import argparse
import os
import sys
import logging
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from GPT_SoVITS.inference_webui import change_gpt_weights, change_sovits_weights, get_tts_wav
import soundfile as sf
import uvicorn
import config as global_config
from tools.i18n.i18n import I18nAuto
import nltk

# Download required NLTK data
nltk.download('averaged_perceptron_tagger_eng')

# Logger configuration
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
logger = logging.getLogger('uvicorn')

# Get configuration
g_config = global_config.Config()
i18n = I18nAuto()

# Language mapping
dict_language = {
    "中文": "all_zh",
    "粤语": "all_yue",
    "英文": "en",
    "日文": "all_ja",
    "韩文": "all_ko",
    "中英混合": "zh",
    "粤英混合": "yue",
    "日英混合": "ja",
    "韩英混合": "ko",
    "多语种混合": "auto",
    "多语种混合(粤语)": "auto_yue",
    "all_zh": "all_zh",
    "all_yue": "all_yue",
    "en": "en",
    "all_ja": "all_ja",
    "all_ko": "all_ko",
    "zh": "zh",
    "yue": "yue",
    "ja": "ja",
    "ko": "ko",
    "auto": "auto",
    "auto_yue": "auto_yue",
}

# Parse arguments
parser = argparse.ArgumentParser(description="GPT-SoVITS API")
parser.add_argument("-s", "--sovits_path", type=str, default=g_config.sovits_path, help="SoVITS model path")
parser.add_argument("-g", "--gpt_path", type=str, default=g_config.gpt_path, help="GPT model path")
parser.add_argument("-dr", "--default_refer_path", type=str, default="", help="Default reference audio path")
parser.add_argument("-dt", "--default_refer_text", type=str, default="", help="Default reference audio text")
parser.add_argument("-dl", "--default_refer_language", type=str, default="", help="Default reference audio language")
parser.add_argument("-d", "--device", type=str, default=g_config.infer_device, help="cuda / cpu")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 0.0.0.0")
parser.add_argument("-p", "--port", type=int, default=g_config.api_port, help="default: 9880")
parser.add_argument("-fp", "--full_precision", action="store_true", default=False, help="Use full precision")
parser.add_argument("-hp", "--half_precision", action="store_true", default=False, help="Use half precision")
parser.add_argument("-sm", "--stream_mode", type=str, default="close", help="Stream mode: close / normal")
parser.add_argument("-mt", "--media_type", type=str, default="wav", help="Audio format: wav / ogg / aac")
parser.add_argument("-st", "--sub_type", type=str, default="int16", help="Audio data type: int16 / int32")
parser.add_argument("-cp", "--cut_punc", type=str, default="", help="Text segmentation symbols")
parser.add_argument("-hb", "--hubert_path", type=str, default=g_config.cnhubert_path, help="CNHubert path")
parser.add_argument("-b", "--bert_path", type=str, default=g_config.bert_path, help="BERT path")

args = parser.parse_args()
sovits_path = args.sovits_path or g_config.pretrained_sovits_path
gpt_path = args.gpt_path or g_config.pretrained_gpt_path
device = args.device
port = args.port
host = args.bind_addr
cnhubert_base_path = args.hubert_path
bert_path = args.bert_path
default_cut_punc = args.cut_punc

# Log model paths
logger.info(f"SoVITS model path: {sovits_path}")
logger.info(f"GPT model path: {gpt_path}")

# Half precision setup
is_half = g_config.is_half
if args.full_precision:
    is_half = False
if args.half_precision:
    is_half = True
if args.full_precision and args.half_precision:
    is_half = g_config.is_half
logger.info(f"Half precision: {is_half}")

# Stream mode and media type
stream_mode = "normal" if args.stream_mode.lower() in ["normal", "n"] else "close"
logger.info(f"Stream mode: {stream_mode}")
media_type = args.media_type.lower() if args.media_type.lower() in ["aac", "ogg"] else "wav" if stream_mode == "close" else "ogg"
logger.info(f"Media type: {media_type}")

# Audio data type
is_int32 = args.sub_type.lower() == "int32"
logger.info(f"Data type: {'int32' if is_int32 else 'int16'}")

# FastAPI app
app = FastAPI()

@app.get("/")
async def tts_endpoint(
    prompt_text: str = "今日は友達と一緒に映画を見に行く予定ですが、天気が悪くて少し心配です。",
    prompt_language: str = "日文",
    character: str = "saotome",
    text: str = None,
    text_language: str = None,
    cut_punc: str = None,
    top_k: int = 15,
    top_p: float = 1.0,
    temperature: float = 1.0,
    speed: float = 1.0,
    sample_steps: int = 20,
    if_sr: bool = False,
    version: str = "v1",
    loudness_boost: str = "false",
    gain: str = "0",
    normalize: str = "false",
    energy_scale: str = "1.0",
    volume_scale: str = "1.0",
    strain_effect: str = "0.0"
):
    try:
        # Character-specific configurations
        character = character.lower()
        ref_wav_path = f"idols/{character}/{character}.wav"
        gpt_path = sovits_path = None
        if character == "kurari":
            prompt_text = "おはよう〜。今日はどんな1日過ごすー？くらりはね〜いつでもあなたの味方だよ"
            gpt_path = "GPT_SoVITS/pretrained_models/kurari-e40.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/kurari_e20_s1800_l32.pth"
            if version == "v2":
                gpt_path = "GPT_SoVITS/pretrained_models/kurari-hql-e40.ckpt"
                sovits_path = "GPT_SoVITS/pretrained_models/kurari-hql_e20_s1240.pth"
            elif version == "v3":
                gpt_path = "GPT_SoVITS/pretrained_models/kurari-high-e45.ckpt"
                sovits_path = "GPT_SoVITS/pretrained_models/kurari-high_e25_s325.pth"
        elif character == "saotome":
            prompt_text = "朝ごはんにはトーストと卵、そしてコーヒーを飲みました。簡単だけど、朝の時間が少し幸せに感じられる瞬間でした。"
            gpt_path = "GPT_SoVITS/pretrained_models/saotome-e30.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/saotome_e9_s522_l32.pth"
        elif character == "ruroro":
            prompt_text = "若是看到自己的朋友改囤原池抽錯或是拿去抽長柱池的話記得把影片分享給他看"
            prompt_language = "中英混合"
            gpt_path = "GPT_SoVITS/pretrained_models/ruroro-e40.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/s2Gv2ProPlus.pth"
        elif character in ["ikko", "ikka"]:
            prompt_text = "せおいなげ、まじばな、らぶらぶ、あげあげ、まぼろし"
            gpt_path = "GPT_SoVITS/pretrained_models/ikko-san-e45.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/s2Gv2ProPlus.pth"
            ref_wav_path = "idols/ikka/ikko_boost.wav" if loudness_boost.lower() == "true" else "idols/ikka/ikko.wav"
        elif character == "baacharu":
            prompt_text = "どーもー、世界初男性バーチャルユーチューバーのばあちゃるです"
            gpt_path = "GPT_SoVITS/pretrained_models/baacharu-e40.ckpt"
            sovits_path = "GPT_SoVITS/pretrained_models/baacharu_e15_s1320_l32.pth"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported character: {character}")

        # Validate inputs
        if not text or not text_language:
            raise HTTPException(status_code=400, detail="text and text_language are required")

        # Convert string parameters
        loudness_boost = loudness_boost.lower() == "true"
        gain = float(gain)
        normalize = normalize.lower() == "true"
        energy_scale = float(energy_scale)
        volume_scale = float(volume_scale)
        strain_effect = float(strain_effect)

        # Map text_language to internal format
        text_language_map = {
            "all_ja": "日文",
            "ja": "日英混合",
            "en": "英文",
            "zh": "中英混合",
            "all_zh": "中文",
            "all_ko": "韩文"
        }
        text_language = text_language_map.get(text_language.lower(), text_language)

        # Log request details
        logger.info(f"Processing TTS request for character: {character}, prompt_language: {prompt_language}, text_language: {text_language}")
        logger.info(f"Loading GPT weights: {gpt_path}")
        try:
            change_gpt_weights(gpt_path=gpt_path)
            logger.info(f"Successfully loaded GPT weights: {gpt_path}")
        except Exception as e:
            logger.error(f"Error loading GPT weights: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load GPT weights: {str(e)}")

        logger.info(f"Loading SoVITS weights: {sovits_path}, prompt_language: {i18n(prompt_language)}, text_language: {i18n(text_language)}")
        try:
            change_sovits_weights(sovits_path=sovits_path, prompt_language=i18n(prompt_language), text_language=i18n(text_language))
            logger.info(f"Successfully loaded SoVITS weights: {sovits_path}")
        except Exception as e:
            logger.error(f"Error loading SoVITS weights: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load SoVITS weights: {str(e)}")

        # Synthesize audio
        synthesis_result = get_tts_wav(
            ref_wav_path=ref_wav_path,
            prompt_text=prompt_text,
            prompt_language=i18n(prompt_language),
            text=text,
            text_language=i18n(text_language),
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
            sample_steps=sample_steps,
            if_sr=if_sr,
            loudness_boost=loudness_boost,
            gain=gain,
            normalize=normalize,
            energy_scale=energy_scale,
            volume_scale=volume_scale,
            strain_effect=strain_effect
        )

        # Process synthesis result
        result_list = list(synthesis_result)
        if not result_list:
            logger.error("Failed to generate audio")
            raise HTTPException(status_code=400, detail="Failed to generate audio")

        last_sampling_rate, last_audio_data = result_list[-1]
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, last_audio_data, last_sampling_rate, format="wav")
        audio_buffer.seek(0)

        logger.info(f"Generated audio with sampling rate: {last_sampling_rate}")
        return StreamingResponse(
            audio_buffer,
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=output.wav"}
        )

    except Exception as e:
        logger.error(f"Error in tts_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting GPT-SoVITS API server")
    uvicorn.run(app, host=host, port=port, workers=1)