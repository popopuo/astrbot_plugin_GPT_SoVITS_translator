import base64
import random

from astrbot.api import logger
from astrbot.api.event import filter
from astrbot.api.star import Context, Star
from astrbot.core import AstrBotConfig
from astrbot.core.message.components import Plain, Record
from astrbot.core.platform import AstrMessageEvent

from .core.client import GSVApiClient, GSVRequestResult
from .core.config import PluginConfig
from .core.emotion import EmotionJudger
from .core.entry import EntryManager
from .core.local_data import LocalDataManager
from .core.service import GPTSoVITSService
from .core.translator import TextTranslator


class GPTSoVITSPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.cfg = PluginConfig(config, context)
        self.local_data = LocalDataManager(self.cfg)
        self.entry_mgr = EntryManager(self.cfg)
        self.client = GSVApiClient(self.cfg)
        self.judger = EmotionJudger(self.cfg)
        self.translator = TextTranslator(self.cfg)
        self.service = GPTSoVITSService(self.cfg, self.client, self.local_data)

    async def initialize(self):
        if self.cfg.enabled:
            await self.service.load_model()

    async def terminate(self):
        await self.client.close()

    @staticmethod
    def _to_record(res: GSVRequestResult) -> Record:
        if res.file_path:
            try:
                return Record.fromFileSystem(res.file_path)
            except Exception:
                logger.warning(f"无法读取文件：{res.file_path}, 已忽略")
                pass

        if not res.data:
            raise ValueError("无法获取结果数据")

        b64 = base64.b64encode(res.data).decode()
        return Record.fromBase64(b64)


    async def _get_emotion_params(
        self, event: AstrMessageEvent, text: str
    ) -> dict | None:
        entry = None

        if self.cfg.judge.enabled_llm:
            labels = self.entry_mgr.get_names()
            emotion = await self.judger.judge_emotion(event, text=text, labels=labels)
            if emotion:
                entry = self.entry_mgr.get_entry(emotion)

        if entry is None:
            entry = self.entry_mgr.match_entry(text)

        return entry.to_params() if entry else None

    @filter.on_decorating_result(priority=14)
    async def on_decorating_result(self, event: AstrMessageEvent):
        """消息入口"""
        if not self.cfg.enabled:
            logger.debug("[auto_tts] skip: plugin disabled")
            return
        cfg = self.cfg.auto

        result = event.get_result()
        if not result:
            logger.debug("[auto_tts] skip: no result")
            return
        chain = result.chain
        if not chain:
            logger.debug("[auto_tts] skip: empty chain")
            return
        if cfg.only_llm_result and not result.is_llm_result():
            logger.debug("[auto_tts] skip: only_llm_result=True and not LLM result")
            return
        if random.random() > cfg.tts_prob:
            logger.debug("[auto_tts] skip: probability not hit")
            return

        # 收集所有Plain文本片段
        plain_texts = []
        for seg in chain:
            if isinstance(seg, Plain):
                plain_texts.append(seg.text)

        # 仅允许只含有Plain的消息链通过
        if len(plain_texts) != len(chain):
            logger.debug("[auto_tts] skip: chain contains non-Plain components")
            return

        # 合并所有Plain文本
        combined_text = "\n".join(plain_texts)

        bypass_cache = False
        translated_ok = False

        if self.cfg.translate.enabled_llm and (not self.cfg.translate.only_llm_tool):
            logger.debug(
                f"[auto_tts] translating to target_lang={self.cfg.translate.target_lang}"
            )
            translated = await self.translator.translate(
                event,
                text=combined_text,
                target_lang=self.cfg.translate.target_lang,
            )
            if translated:
                combined_text = translated
                translated_ok = True
                logger.debug(f"[auto_tts] translation ok, len={len(combined_text)}")
            else:
                logger.debug("[auto_tts] translation skipped/failed, fallback to original")
            bypass_cache = True

        # 仅允许一定长度以下的文本通过
        if len(combined_text) > cfg.max_msg_len:
            logger.debug("[auto_tts] skip: text too long")
            return

        params = await self._get_emotion_params(event, combined_text)

        if (
            self.cfg.translate.enabled_llm
            and (not self.cfg.translate.only_llm_tool)
            and translated_ok
            and self.cfg.translate.target_lang in {"zh", "en", "ja", "ko"}
        ):
            params = params.copy() if params else {}
            params["text_lang"] = self.cfg.translate.target_lang

        res = await self.service.inference(
            combined_text,
            extra_params=params,
            use_cache=not bypass_cache,
        )
        if not bool(res):
            logger.debug("[auto_tts] skip: TTS inference failed")
            return
        chain.append(self._to_record(res))
        logger.debug("[auto_tts] TTS appended to chain")

    @filter.command("说", alias={"gsv", "GSV"})
    async def on_command(self, event: AstrMessageEvent):
        """说 <内容>, 直接调用GSV合成语音"""
        if not self.cfg.enabled:
            return

        original_text = event.message_str.partition(" ")[2]

        text = original_text
        bypass_cache = False
        translated_ok = False

        if self.cfg.translate.enabled_llm and (not self.cfg.translate.only_llm_tool):
            logger.debug(
                f"[say] translating to target_lang={self.cfg.translate.target_lang}"
            )
            translated = await self.translator.translate(
                event,
                text=text,
                target_lang=self.cfg.translate.target_lang,
            )
            if translated:
                text = translated
                translated_ok = True
                logger.debug(f"[say] translation ok, len={len(text)}")
            else:
                logger.debug("[say] translation skipped/failed, fallback to original")
            bypass_cache = True

        params = await self._get_emotion_params(event, text)

        if (
            self.cfg.translate.enabled_llm
            and (not self.cfg.translate.only_llm_tool)
            and translated_ok
            and self.cfg.translate.target_lang in {"zh", "en", "ja", "ko"}
        ):
            params = params.copy() if params else {}
            params["text_lang"] = self.cfg.translate.target_lang

        res = await self.service.inference(text, extra_params=params, use_cache=not bypass_cache)

        if not bool(res):
            yield event.plain_result(res.error)
            return

        yield event.chain_result([Plain(original_text), self._to_record(res)])

    @filter.command("重启GSV", alias={"重启gsv"})
    async def tts_control(self, event: AstrMessageEvent):
        """重启GPT_SoVITS"""
        if not self.cfg.enabled:
            return
        yield event.plain_result("重启TTS中...(报错信息请忽略，等待一会即可完成重启)")
        await self.service.restart()

    @filter.llm_tool()
    async def gsv_tts(self, event: AstrMessageEvent, message: str = ""):
        """
        用语音输出要讲的话
        Args:
            message(string): 要讲的话
        """
        try:
            # 当启用“日常自动TTS（非 tool 模式）”时，禁用该 tool，避免 Agent 额外调用导致重复语音。
            if self.cfg.translate.enabled_llm and (not self.cfg.translate.only_llm_tool):
                logger.debug(
                    "[gsv_tts] skipped: auto TTS mode enabled (only_llm_tool=False)"
                )
                return "gsv_tts disabled in auto TTS mode"

            text = message

            logger.debug(f"[gsv_tts] tool called, len={len(text) if text else 0}")

            bypass_cache = False
            translated_ok = False

            if self.cfg.translate.enabled_llm and self.cfg.translate.only_llm_tool:
                logger.debug(
                    f"[gsv_tts] translating to target_lang={self.cfg.translate.target_lang}"
                )
                translated = await self.translator.translate(
                    event,
                    text=text,
                    target_lang=self.cfg.translate.target_lang,
                )
                if translated:
                    text = translated
                    translated_ok = True
                    logger.debug(f"[gsv_tts] translation ok, len={len(text)}")
                else:
                    logger.debug("[gsv_tts] translation skipped/failed, fallback to original")

                bypass_cache = True

            params = await self._get_emotion_params(event, text)

            if (
                self.cfg.translate.enabled_llm
                and self.cfg.translate.only_llm_tool
                and translated_ok
                and self.cfg.translate.target_lang in {"zh", "en", "ja", "ko"}
            ):
                params = params.copy() if params else {}
                params["text_lang"] = self.cfg.translate.target_lang

            logger.debug(f"[gsv_tts] calling tts, len={len(text) if text else 0}")
            res = await self.service.inference(
                text,
                extra_params=params,
                use_cache=not bypass_cache,
            )
            if not bool(res):
                return res.error
            seg = self._to_record(res)
            await event.send(event.chain_result([seg]))
        except Exception as e:
            return str(e)
