import json

from astrbot.api import logger
from astrbot.core.platform import AstrMessageEvent

from .config import PluginConfig


class TextTranslator:
    def __init__(self, config: PluginConfig):
        self.cfg = config

    async def translate(
        self,
        event: AstrMessageEvent,
        *,
        text: str,
        target_lang: str,
    ) -> str | None:
        """使用 LLM 翻译文本。

        对外行为约定：
        - 本方法不会抛出异常，失败返回 None
        - 会按 (target_lang, text) 做 event.extra 级缓存，避免同一事件重复调用
        """
        if not text or not target_lang:
            return None

        cache_key = f"translate:{target_lang}:{hash(text)}"
        cached = event.get_extra(cache_key)
        if cached and isinstance(cached, str):
            return cached

        try:
            provider = self.cfg.get_translate_provider(event.unified_msg_origin)
            system_prompt, prompt = self._build_prompt(text=text, target_lang=target_lang)

            resp = await provider.text_chat(
                system_prompt=system_prompt,
                prompt=prompt,
            )

            translated = self._parse_llm_response(resp.completion_text)
            if not translated:
                return None

            event.set_extra(cache_key, translated)
            return translated

        except Exception as e:
            logger.exception(f"翻译失败: {e}")
            return None

    def _build_prompt(self, *, text: str, target_lang: str) -> tuple[str, str]:
        system_prompt = (
            "你是一个专业翻译。\n"
            "请将用户提供的文本翻译成目标语言。\n"
            "要求：\n"
            "1) 只输出翻译后的文本，不要包含任何多余内容（不要解释/不要前后缀/不要代码块）。\n"
            "2) 保留原文中的换行与语气（尽量自然）。\n"
            "3) 遇到人名/专有名词尽量保持一致。\n"
        )
        prompt = f"目标语言：{target_lang}\n原文：{text}"
        return system_prompt, prompt

    def _parse_llm_response(self, text: str) -> str:
        if not text:
            return ""

        raw = text.strip()

        # 1) 优先按严格 JSON 解析
        try:
            data = json.loads(raw)
            translated = data.get("text") if isinstance(data, dict) else None
            if translated and isinstance(translated, str):
                return translated.strip()
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"解析翻译 JSON 失败，将尝试按纯文本处理: {e}")

        # 2) 兼容模型输出 ```json ... ``` / ``` ... ``` 代码块
        if raw.startswith("```"):
            lines = raw.splitlines()
            if len(lines) >= 2 and lines[0].startswith("```"):
                # 去掉首尾 fence
                end_idx = None
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i].strip().startswith("```"):
                        end_idx = i
                        break
                if end_idx is not None and end_idx > 0:
                    raw = "\n".join(lines[1:end_idx]).strip()

        # 3) 再尝试一次 JSON（有些模型会把 JSON 包在代码块里）
        try:
            data = json.loads(raw)
            translated = data.get("text") if isinstance(data, dict) else None
            if translated and isinstance(translated, str):
                return translated.strip()
        except Exception:
            pass

        # 4) 最后回退：把整个输出当翻译结果
        logger.debug(f"翻译返回非 JSON，按纯文本采用: {raw[:200]}")
        return raw
