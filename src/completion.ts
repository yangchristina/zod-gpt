import {
  TokenError,
  CompletionApi,
  AnthropicChatApi,
  ChatRequestMessage,
  ChatResponse,
  ModelRequestOptions,
} from 'llm-api';
import { defaults } from 'lodash';
import { z } from 'zod';

import type { RequestOptions, Response, ResponseZod } from './types';
import { debug, parseUnsafeJson, zodToJsonSchema } from './utils';

const FunctionName = 'print';
const FunctionDescription =
  'ALWAYS respond by calling this function with the given parameters';

const Defaults = {
  autoHeal: true,
  autoSlice: false,
};

function respondWrapper<T extends z.ZodType = z.ZodString>(
  respond: (
    message: string | ChatRequestMessage,
    opt?: ModelRequestOptions | undefined,
  ) => Promise<ChatResponse>,
  prompt: string | (() => string),
  _opt: Partial<RequestOptions<T>> | undefined,
  isAnthropic: boolean,
) {
  const { message, schemaInstructions } = handleRequest(
    prompt,
    _opt,
    isAnthropic,
  );

  // const response = await completionLogic(model, prompt, _opt);
  return async (
    prompt: string | (() => string),
    _override_opt?: Partial<RequestOptions<T>> | undefined,
  ) => {
    const _opt2 = {
      ..._opt,
      ..._override_opt,
    };
    const initialResponse = await respond(message, _opt2);
    const response = await handleResponse(
      initialResponse,
      schemaInstructions,
      _opt2,
      isAnthropic,
    );
    // Warning: not handling slicing
    return {
      ...response,
      respond: respondWrapper(response.respond, prompt, _opt2, isAnthropic),
    };
  };
}

export async function completion<T extends z.ZodType = z.ZodString>(
  model: CompletionApi,
  prompt: string | (() => string),
  _opt?: Partial<RequestOptions<T>>,
): Promise<ResponseZod<T>> {
  const isAnthropic = model instanceof AnthropicChatApi;
  const response = await completionLogic(model, prompt, _opt);
  return {
    ...response,
    respond: respondWrapper(response.respond, prompt, _opt, isAnthropic),
  };
}

function handleRequest<T extends z.ZodType = z.ZodString>(
  prompt: string | (() => string),
  _opt: Partial<RequestOptions<T>> | undefined,
  isAnthropic: boolean,
) {
  const message = typeof prompt === 'string' ? prompt : prompt();
  const jsonSchema = _opt?.schema && zodToJsonSchema(_opt?.schema);
  const opt = defaults(
    {
      // build function to call if schema is defined
      callFunction: _opt?.schema ? FunctionName : undefined,
      functions: _opt?.schema
        ? [
            {
              name: FunctionName,
              description: FunctionDescription,
              parameters: jsonSchema,
            },
          ]
        : undefined,
    },
    _opt,
    Defaults,
  );

  if (
    opt.schema &&
    (opt.schema._def as any).typeName !== z.ZodFirstPartyTypeKind.ZodObject
  ) {
    throw new Error('Schemas can ONLY be an object');
  }
  debug.log('⬆️ sending request:', message);

  const schemaInstructions =
    isAnthropic && _opt?.schema && JSON.stringify(jsonSchema);
  const firstSchemaKey =
    isAnthropic && _opt?.schema && Object.keys(jsonSchema['properties'])[0];
  const responsePrefix = `{ "${firstSchemaKey}": `;

  return {
    message,
    opt,
    schemaInstructions,
    responsePrefix,
    requestOptions: {
      ...opt,
      systemMessage:
        `You will respond to ALL human messages in JSON. Make sure the response correctly follow the following JSON schema specifications: ${schemaInstructions}\n\n${
          opt.systemMessage
            ? typeof opt.systemMessage === 'string'
              ? opt.systemMessage
              : opt.systemMessage()
            : ''
        }`.trim(),
      responsePrefix,
    },
  };
}

async function completionLogic<T extends z.ZodType = z.ZodString>(
  model: CompletionApi,
  prompt: string | (() => string),
  _opt?: Partial<RequestOptions<T>>,
): Promise<Response<T>> {
  const isAnthropic = model instanceof AnthropicChatApi;
  const message = typeof prompt === 'string' ? prompt : prompt();
  const jsonSchema = _opt?.schema && zodToJsonSchema(_opt?.schema);
  const opt = defaults(
    {
      // build function to call if schema is defined
      callFunction: _opt?.schema ? FunctionName : undefined,
      functions: _opt?.schema
        ? [
            {
              name: FunctionName,
              description: FunctionDescription,
              parameters: jsonSchema,
            },
          ]
        : undefined,
    },
    _opt,
    Defaults,
  );

  try {
    const { schemaInstructions, requestOptions } = handleRequest(
      prompt,
      _opt,
      isAnthropic,
    );
    const messages: ChatRequestMessage[] = [
      ...(opt.messageHistory ?? []),
      { role: 'user', content: message },
    ];

    // Anthropic does not have support for functions, so create a custom system message and inject it as the first system message
    // Use the `responsePrefix` property to steer anthropic to output in the json structure
    const response =
      isAnthropic && _opt?.schema
        ? await model.chatCompletion(messages, requestOptions)
        : await model.chatCompletion(messages, opt);
    if (!response) {
      throw new Error('Chat request failed');
    }

    // only send this debug msg when stream is not enabled, or there'll be duplicate log msgs since stream also streams in the logs
    !model.modelConfig.stream && debug.log('⬇️ received response:', response);

    return await handleResponse(response, schemaInstructions, opt, isAnthropic);
  } catch (e) {
    if (e instanceof TokenError && opt.autoSlice) {
      const chunkSize = message.length - e.overflowTokens * 4;
      if (chunkSize < 0) {
        throw e;
      }

      debug.log(
        `⚠️ Request prompt too long, splitting text with chunk size of ${chunkSize}`,
      );
      return completionLogic(model, message.slice(0, chunkSize), opt);
    } else {
      throw e;
    }
  }
}

async function handleResponse<T extends z.ZodType = z.ZodString>(
  response: ChatResponse,
  schemaInstructions: string | false | undefined,
  opt?: Partial<RequestOptions<T>>,
  isAnthropic = false,
) {
  if (!response) {
    throw new Error('Chat request failed');
  }

  // validate res content, and recursively loop if invalid
  if (opt?.schema) {
    if (!isAnthropic && !response.arguments) {
      if (opt.autoHeal) {
        debug.log('⚠️ function not called, autohealing...');
        response = await response.respond({
          role: 'user',
          content: `Please respond with a call to the ${FunctionName} function`,
        });

        if (!response.arguments) {
          throw new Error('Response function autoheal failed');
        }
      } else {
        throw new Error('Response function not called');
      }
    }

    let json = isAnthropic
      ? parseUnsafeJson(response.content ?? '')
      : response.arguments;
    if (!json) {
      throw new Error('No response received');
    }

    const res = opt.schema.safeParse(json);
    if (res.success) {
      return {
        ...response,
        data: res.data,
      };
    } else {
      debug.error('⚠️ error parsing response', res.error);
      if (opt.autoHeal) {
        debug.log('⚠️ response parsing failed, autohealing...', res.error);
        const issuesMessage = res.error.issues.reduce(
          (prev, issue) =>
            issue.path && issue.path.length > 0
              ? `${prev}\nThe issue is at path ${issue.path.join('.')}: ${
                  issue.message
                }.`
              : `\nThe issue is: ${issue.message}.`,
          isAnthropic
            ? `There is an issue with that response, please follow the JSON schema EXACTLY, the output must be valid parsable JSON: ${schemaInstructions}`
            : `There is an issue with that response, please rewrite by calling the ${FunctionName} function with the correct parameters.`,
        );
        response = await response.respond(issuesMessage);
      } else {
        throw new Error('Response parsing failed');
      }
    }

    json = isAnthropic
      ? parseUnsafeJson(response.content ?? '')
      : response.arguments;
    if (!json) {
      throw new Error('Response schema autoheal failed');
    }

    // TODO: there is definitely a cleaner way to implement this to avoid the duplicate parsing
    const data = opt.schema.parse(json);
    return {
      ...response,
      data,
    };
  }

  // if no schema is defined, default to string
  return {
    ...response,
    data: String(response.content),
  };
}
