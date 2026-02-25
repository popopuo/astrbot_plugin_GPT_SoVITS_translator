import json

from astrbot.api import logger
from astrbot.core.platform.astr_message_event import AstrMessageEvent

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
            "1) 只输出 JSON，不要包含任何多余内容（不要代码块）。\n"
            "2) 仅翻译，不要解释，不要添加前后缀。\n"
            "3) 保留原文中的换行与语气（尽量自然）。\n"
            '输出示例：{"text":"..."}'
        )
        prompt = f"目标语言：{target_lang}\n原文：{text}"
        return system_prompt, prompt

    def _parse_llm_response(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM 返回非 JSON: {text}") from e

        translated = data.get("text")
        if not translated or not isinstance(translated, str):
            raise ValueError(f"LLM JSON 缺少或非法 text 字段: {data}")

        return translated.strip()
