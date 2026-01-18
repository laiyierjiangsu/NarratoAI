"""
LiteLLM 统一提供商实现

使用 LiteLLM 库提供统一的 LLM 接口，支持 100+ providers
包括 OpenAI, Anthropic, Gemini, Qwen, DeepSeek, SiliconFlow 等
"""

import asyncio
import base64
import io
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import PIL.Image
from loguru import logger

try:
    import litellm
    from litellm import acompletion, completion
    from litellm.exceptions import (
        AuthenticationError as LiteLLMAuthError,
        RateLimitError as LiteLLMRateLimitError,
        BadRequestError as LiteLLMBadRequestError,
        APIError as LiteLLMAPIError
    )
except ImportError:
    logger.error("LiteLLM 未安装。请运行: pip install litellm")
    raise

from .base import VisionModelProvider, TextModelProvider
from .exceptions import (
    APICallError,
    AuthenticationError,
    RateLimitError,
    ContentFilterError
)


# 配置 LiteLLM 全局设置
def configure_litellm():
    """配置 LiteLLM 全局参数"""
    from app.config import config

    # 设置重试次数
    litellm.num_retries = config.app.get('llm_max_retries', 3)

    # 设置默认超时
    litellm.request_timeout = config.app.get('llm_text_timeout', 180)

    # 启用详细日志（开发环境）
    # litellm.set_verbose = True

    logger.info(f"LiteLLM 配置完成: retries={litellm.num_retries}, timeout={litellm.request_timeout}s")


# 初始化配置
configure_litellm()


class LiteLLMVisionProvider(VisionModelProvider):
    """使用 LiteLLM 的统一视觉模型提供商"""

    @property
    def provider_name(self) -> str:
        # 从 model_name 中提取 provider 名称（如 "gemini/gemini-2.0-flash"）
        if "/" in self.model_name:
            return self.model_name.split("/")[0]
        if getattr(self, "base_url", None) and "openrouter" in self.base_url.lower():
            return "openrouter"
        return "litellm"

    @property
    def supported_models(self) -> List[str]:
        # LiteLLM 支持 100+ providers 和数百个模型，无法全部列举
        # 返回空列表表示跳过预定义列表检查，由 LiteLLM 在实际调用时验证
        return []

    def _validate_model_support(self):
        """
        重写模型验证逻辑

        对于 LiteLLM，我们不做预定义列表检查，因为：
        1. LiteLLM 支持 100+ providers 和数百个模型，无法全部列举
        2. LiteLLM 会在实际调用时进行模型验证
        3. 如果模型不支持，LiteLLM 会返回清晰的错误信息

        这里只做基本的格式验证（可选）
        """
        from loguru import logger

        # 可选：检查模型名称格式（provider/model）
        if "/" not in self.model_name:
            logger.debug(
                f"LiteLLM 模型名称 '{self.model_name}' 未包含 provider 前缀，"
                f"LiteLLM 将尝试自动推断。建议使用 'provider/model' 格式，如 'gemini/gemini-2.5-flash'"
            )

        # 不抛出异常，让 LiteLLM 在实际调用时验证
        logger.debug(f"LiteLLM 视觉模型已配置: {self.model_name}")

    def _initialize(self):
        """初始化 LiteLLM 特定设置"""
        # 设置 API key 到环境变量（LiteLLM 会自动读取）
        import os

        # 根据 model_name 确定需要设置哪个 API key
        provider = self.provider_name.lower()

        # 映射 provider 到环境变量名
        env_key_mapping = {
            "gemini": "GEMINI_API_KEY",
            "google": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "qwen": "QWEN_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "openrouter": "OPENAI_API_KEY"  # OpenRouter uses OpenAI-compatible API
        }

        env_var = env_key_mapping.get(provider, f"{provider.upper()}_API_KEY")

        if self.api_key and env_var:
            os.environ[env_var] = self.api_key
            logger.debug(f"设置环境变量: {env_var}")

        # 如果提供了 base_url，设置到 LiteLLM
        if self.base_url:
            # LiteLLM 支持通过 api_base 参数设置自定义 URL
            self._api_base = self.base_url
            logger.debug(f"使用自定义 API base URL: {self.base_url}")
        
        # 保存 api_key 用于直接传递（某些 provider 需要）
        self._api_key = self.api_key

    async def analyze_images(self,
                           images: List[Union[str, Path, PIL.Image.Image]],
                           prompt: str,
                           batch_size: int = 10,
                           **kwargs) -> List[str]:
        """
        使用 LiteLLM 分析图片

        Args:
            images: 图片路径列表或PIL图片对象列表
            prompt: 分析提示词
            batch_size: 批处理大小
            **kwargs: 其他参数

        Returns:
            分析结果列表
        """
        logger.info(f"开始使用 LiteLLM ({self.model_name}) 分析 {len(images)} 张图片")

        # 预处理图片
        processed_images = self._prepare_images(images)

        # 分批处理
        results = []
        for i in range(0, len(processed_images), batch_size):
            batch = processed_images[i:i + batch_size]
            logger.info(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 张图片")

            try:
                result = await self._analyze_batch(batch, prompt, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"批次 {i//batch_size + 1} 处理失败: {str(e)}")
                results.append(f"批次处理失败: {str(e)}")

        return results

    async def _analyze_batch(self, batch: List[PIL.Image.Image], prompt: str, **kwargs) -> str:
        """分析一批图片"""
        # 构建 LiteLLM 格式的消息
        content = [{"type": "text", "text": prompt}]

        # 添加图片（使用 base64 编码）
        for img in batch:
            base64_image = self._image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            })

        messages = [{
            "role": "user",
            "content": content
        }]

        # 调用 LiteLLM
        try:
            # 准备参数
            effective_model_name = self.model_name
            
            # OpenRouter 特殊处理
            is_openrouter = self.provider_name.lower() == "openrouter"
            if is_openrouter:
                # OpenRouter 使用 OpenAI 兼容 API，但模型名称需要保持 openrouter/ 前缀
                # LiteLLM 会自动识别 openrouter/ 前缀并正确处理
                if self.model_name.lower().startswith("openrouter/"):
                    effective_model_name = self.model_name
                else:
                    effective_model_name = f"openrouter/{self.model_name}"
                logger.debug(f"使用 OpenRouter 视觉模型: {effective_model_name}")
            
            # SiliconFlow 特殊处理
            elif self.model_name.lower().startswith("siliconflow/"):
                # 替换 provider 为 openai
                if "/" in self.model_name:
                    effective_model_name = f"openai/{self.model_name.split('/', 1)[1]}"
                else:
                    effective_model_name = f"openai/{self.model_name}"
                
                # 确保设置了 OPENAI_API_KEY (如果尚未设置)
                import os
                if not os.environ.get("OPENAI_API_KEY") and os.environ.get("SILICONFLOW_API_KEY"):
                    os.environ["OPENAI_API_KEY"] = os.environ.get("SILICONFLOW_API_KEY")
                    
                # 确保设置了 base_url (如果尚未设置)
                if not hasattr(self, '_api_base'):
                     self._api_base = "https://api.siliconflow.cn/v1"

            completion_kwargs = {
                "model": effective_model_name,
                "messages": messages,
                "temperature": kwargs.get("temperature", 1.0),
                "max_tokens": kwargs.get("max_tokens", 4000)
            }

            # 如果有自定义 base_url，添加 api_base 参数
            if hasattr(self, '_api_base'):
                completion_kwargs["api_base"] = self._api_base

            # 支持动态传递 api_key 和 api_base
            if "api_key" in kwargs:
                completion_kwargs["api_key"] = kwargs["api_key"]
            elif hasattr(self, '_api_key') and self._api_key:
                # 如果没有通过 kwargs 传递，使用实例的 api_key
                # 这对于 OpenRouter 等需要直接传递 api_key 的 provider 很重要
                completion_kwargs["api_key"] = self._api_key
            if "api_base" in kwargs:
                completion_kwargs["api_base"] = kwargs["api_base"]

            response = await acompletion(**completion_kwargs)

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                logger.debug(f"LiteLLM 调用成功，消耗 tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
                return content
            else:
                raise APICallError("LiteLLM 返回空响应")

        except LiteLLMAuthError as e:
            logger.error(f"LiteLLM 认证失败: {str(e)}")
            raise AuthenticationError()
        except LiteLLMRateLimitError as e:
            logger.error(f"LiteLLM 速率限制: {str(e)}")
            raise RateLimitError()
        except LiteLLMBadRequestError as e:
            error_msg = str(e)
            if "SAFETY" in error_msg.upper() or "content_filter" in error_msg.lower():
                raise ContentFilterError(f"内容被安全过滤器阻止: {error_msg}")
            logger.error(f"LiteLLM 请求错误: {error_msg}")
            raise APICallError(f"请求错误: {error_msg}")
        except LiteLLMAPIError as e:
            logger.error(f"LiteLLM API 错误: {str(e)}")
            raise APICallError(f"API 错误: {str(e)}")
        except Exception as e:
            logger.error(f"LiteLLM 调用失败: {str(e)}")
            raise APICallError(f"调用失败: {str(e)}")

    def _image_to_base64(self, img: PIL.Image.Image) -> str:
        """将PIL图片转换为base64编码"""
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='JPEG', quality=85)
        img_bytes = img_buffer.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """兼容基类接口（实际使用 LiteLLM SDK）"""
        pass


class LiteLLMTextProvider(TextModelProvider):
    """使用 LiteLLM 的统一文本生成提供商"""

    @property
    def provider_name(self) -> str:
        # 从 model_name 中提取 provider 名称
        if "/" in self.model_name:
            return self.model_name.split("/")[0]
        if getattr(self, "base_url", None) and "openrouter" in self.base_url.lower():
            return "openrouter"
        # 尝试从模型名称推断 provider
        model_lower = self.model_name.lower()
        if "gpt" in model_lower:
            return "openai"
        elif "claude" in model_lower:
            return "anthropic"
        elif "gemini" in model_lower:
            return "gemini"
        elif "qwen" in model_lower:
            return "qwen"
        elif "deepseek" in model_lower:
            return "deepseek"
        return "litellm"

    @property
    def supported_models(self) -> List[str]:
        # LiteLLM 支持 100+ providers 和数百个模型，无法全部列举
        # 返回空列表表示跳过预定义列表检查，由 LiteLLM 在实际调用时验证
        return []

    def _validate_model_support(self):
        """
        重写模型验证逻辑

        对于 LiteLLM，我们不做预定义列表检查，因为：
        1. LiteLLM 支持 100+ providers 和数百个模型，无法全部列举
        2. LiteLLM 会在实际调用时进行模型验证
        3. 如果模型不支持，LiteLLM 会返回清晰的错误信息

        这里只做基本的格式验证（可选）
        """
        from loguru import logger

        # 可选：检查模型名称格式（provider/model）
        if "/" not in self.model_name:
            logger.debug(
                f"LiteLLM 模型名称 '{self.model_name}' 未包含 provider 前缀，"
                f"LiteLLM 将尝试自动推断。建议使用 'provider/model' 格式，如 'gemini/gemini-2.5-flash'"
            )

        # 不抛出异常，让 LiteLLM 在实际调用时验证
        logger.debug(f"LiteLLM 文本模型已配置: {self.model_name}")

    def _initialize(self):
        """初始化 LiteLLM 特定设置"""
        import os

        # 根据 model_name 确定需要设置哪个 API key
        provider = self.provider_name.lower()

        # 映射 provider 到环境变量名
        env_key_mapping = {
            "gemini": "GEMINI_API_KEY",
            "google": "GEMINI_API_KEY",
            "openai": "OPENAI_API_KEY",
            "qwen": "QWEN_API_KEY",
            "dashscope": "DASHSCOPE_API_KEY",
            "siliconflow": "SILICONFLOW_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "claude": "ANTHROPIC_API_KEY",
            "moonshot": "MOONSHOT_API_KEY",
            "openrouter": "OPENAI_API_KEY"  # OpenRouter uses OpenAI-compatible API
        }

        env_var = env_key_mapping.get(provider, f"{provider.upper()}_API_KEY")

        if self.api_key and env_var:
            os.environ[env_var] = self.api_key
            logger.debug(f"设置环境变量: {env_var}")

        # 如果提供了 base_url，保存用于后续调用
        if self.base_url:
            self._api_base = self.base_url
            logger.debug(f"使用自定义 API base URL: {self.base_url}")
        
        # 保存 api_key 用于直接传递（某些 provider 需要）
        self._api_key = self.api_key

    async def generate_text(self,
                          prompt: str,
                          system_prompt: Optional[str] = None,
                          temperature: float = 1.0,
                          max_tokens: Optional[int] = None,
                          response_format: Optional[str] = None,
                          **kwargs) -> str:
        """
        使用 LiteLLM 生成文本

        Args:
            prompt: 用户提示词
            system_prompt: 系统提示词
            temperature: 生成温度
            max_tokens: 最大token数
            response_format: 响应格式 ('json' 或 None)
            **kwargs: 其他参数

        Returns:
            生成的文本内容
        """
        # 构建消息列表
        messages = self._build_messages(prompt, system_prompt)

        # 准备参数
        effective_model_name = self.model_name
        
        # OpenRouter 特殊处理
        is_openrouter = self.provider_name.lower() == "openrouter"
        if is_openrouter:
            # OpenRouter 使用 OpenAI 兼容 API，但模型名称需要保持 openrouter/ 前缀
            # LiteLLM 会自动识别 openrouter/ 前缀并正确处理
            if self.model_name.lower().startswith("openrouter/"):
                effective_model_name = self.model_name
            else:
                effective_model_name = f"openrouter/{self.model_name}"
            logger.debug(f"使用 OpenRouter 模型: {effective_model_name}")
        
        # SiliconFlow 特殊处理
        elif self.model_name.lower().startswith("siliconflow/"):
            # 替换 provider 为 openai
            if "/" in self.model_name:
                effective_model_name = f"openai/{self.model_name.split('/', 1)[1]}"
            else:
                effective_model_name = f"openai/{self.model_name}"
            
            # 确保设置了 OPENAI_API_KEY (如果尚未设置)
            import os
            if not os.environ.get("OPENAI_API_KEY") and os.environ.get("SILICONFLOW_API_KEY"):
                os.environ["OPENAI_API_KEY"] = os.environ.get("SILICONFLOW_API_KEY")
                
            # 确保设置了 base_url (如果尚未设置)
            if not hasattr(self, '_api_base'):
                    self._api_base = "https://api.siliconflow.cn/v1"

        completion_kwargs = {
            "model": effective_model_name,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            completion_kwargs["max_tokens"] = max_tokens

        # 处理 JSON 格式输出
        # LiteLLM 会自动处理不同 provider 的 JSON mode 差异
        if response_format == "json":
            try:
                completion_kwargs["response_format"] = {"type": "json_object"}
            except Exception as e:
                # 如果不支持，在提示词中添加约束
                logger.warning(f"模型可能不支持 response_format，将在提示词中添加 JSON 约束: {str(e)}")
                messages[-1]["content"] += "\n\n请确保输出严格的JSON格式，不要包含任何其他文字或标记。"

        # OpenRouter 特殊处理：需要设置 api_base 和 api_key
        if is_openrouter:
            # OpenRouter 必须设置 api_base
            if hasattr(self, '_api_base') and self._api_base:
                completion_kwargs["api_base"] = self._api_base
            elif "api_base" not in kwargs:
                # 如果没有设置，使用默认的 OpenRouter API URL
                completion_kwargs["api_base"] = "https://openrouter.ai/api/v1"
            
            # OpenRouter 必须设置 api_key
            if "api_key" in kwargs:
                completion_kwargs["api_key"] = kwargs["api_key"]
            elif hasattr(self, '_api_key') and self._api_key:
                completion_kwargs["api_key"] = self._api_key
            
            logger.debug(f"OpenRouter 配置: api_base={completion_kwargs.get('api_base')}, model={effective_model_name}")
        else:
            # 其他 provider 的处理
            if hasattr(self, '_api_base'):
                completion_kwargs["api_base"] = self._api_base

            # 支持动态传递 api_key 和 api_base
            if "api_key" in kwargs:
                completion_kwargs["api_key"] = kwargs["api_key"]
            elif hasattr(self, '_api_key') and self._api_key:
                completion_kwargs["api_key"] = self._api_key
            if "api_base" in kwargs:
                completion_kwargs["api_base"] = kwargs["api_base"]

        try:
            # 调用 LiteLLM（自动重试）
            response = await acompletion(**completion_kwargs)

            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content

                # 清理可能的 markdown 代码块（针对不支持 JSON mode 的模型）
                if response_format == "json" and "response_format" not in completion_kwargs:
                    content = self._clean_json_output(content)

                logger.debug(f"LiteLLM 调用成功，消耗 tokens: {response.usage.total_tokens if response.usage else 'N/A'}")
                return content
            else:
                raise APICallError("LiteLLM 返回空响应")

        except LiteLLMAuthError as e:
            logger.error(f"LiteLLM 认证失败: {str(e)}")
            raise AuthenticationError()
        except LiteLLMRateLimitError as e:
            logger.error(f"LiteLLM 速率限制: {str(e)}")
            raise RateLimitError()
        except LiteLLMBadRequestError as e:
            error_msg = str(e)
            logger.error(f"LiteLLM BadRequest 错误: {error_msg}")
            
            # 处理不支持 response_format 或参数无效的情况
            # 检查是否是参数相关错误（包括 response_format、Invalid API parameter 等）
            is_param_error = (
                "response_format" in error_msg.lower() or 
                "invalid api parameter" in error_msg.lower() or
                "parameter" in error_msg.lower() and "invalid" in error_msg.lower() or
                "unsupported" in error_msg.lower() and "parameter" in error_msg.lower()
            )
            
            # 对于 OpenRouter，特别处理参数错误
            if is_openrouter:
                logger.warning(f"OpenRouter 模型参数错误，可能是模型不支持某些参数: {error_msg[:300]}")
            
            if is_param_error and response_format == "json" and "response_format" in completion_kwargs:
                logger.warning(f"模型不支持 response_format 参数，重试不带格式约束的请求: {error_msg[:200]}")
                completion_kwargs.pop("response_format", None)
                messages[-1]["content"] += "\n\n请确保输出严格的JSON格式，不要包含任何其他文字或标记。"

                # 重试
                try:
                    response = await acompletion(**completion_kwargs)
                    if response.choices and len(response.choices) > 0:
                        content = response.choices[0].message.content
                        content = self._clean_json_output(content)
                        return content
                    else:
                        raise APICallError("LiteLLM 返回空响应")
                except Exception as retry_error:
                    logger.error(f"重试后仍然失败: {str(retry_error)}")
                    raise APICallError(f"请求错误（已尝试移除不支持的参数）: {str(retry_error)}")

            # 检查是否是安全过滤
            if "SAFETY" in error_msg.upper() or "content_filter" in error_msg.lower():
                raise ContentFilterError(f"内容被安全过滤器阻止: {error_msg}")

            logger.error(f"LiteLLM 请求错误: {error_msg}")
            raise APICallError(f"请求错误: {error_msg}")
        except LiteLLMAPIError as e:
            logger.error(f"LiteLLM API 错误: {str(e)}")
            raise APICallError(f"API 错误: {str(e)}")
        except Exception as e:
            logger.error(f"LiteLLM 调用失败: {str(e)}")
            raise APICallError(f"调用失败: {str(e)}")

    def _clean_json_output(self, output: str) -> str:
        """清理JSON输出，移除markdown标记等"""
        import re

        # 移除可能的markdown代码块标记
        output = re.sub(r'^```json\s*', '', output, flags=re.MULTILINE)
        output = re.sub(r'^```\s*$', '', output, flags=re.MULTILINE)
        output = re.sub(r'^```.*$', '', output, flags=re.MULTILINE)

        # 移除前后空白字符
        output = output.strip()

        return output

    async def _make_api_call(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """兼容基类接口（实际使用 LiteLLM SDK）"""
        pass
