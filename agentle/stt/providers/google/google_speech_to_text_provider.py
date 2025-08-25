from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Mapping, MutableSequence, Sequence
from pathlib import Path
from textwrap import dedent

from rsb.functions.ext2mime import ext2mime

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.generations.providers.failover.failover_generation_provider import (
    FailoverGenerationProvider,
)
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.prompts.models.prompt import Prompt
from agentle.stt.models.audio_transcription import AudioTranscription
from agentle.stt.models.sentence_segment import SentenceSegment
from agentle.stt.models.subtitles import Subtitles
from agentle.stt.models.transcription_config import TranscriptionConfig
from agentle.stt.providers.base.speech_to_text_provider import SpeechToTextProvider

_LANGUAGE_TO_PROMPTS: Mapping[str, Prompt] = {
    # English
    "en": Prompt.from_text(
        dedent("""\
            You are a helpful assistant that transcribes audio files.
            The audio file is {{audio_file}}.
            The language of the audio file is English.
            
            Return ONLY the transcription in SRT format.
            Write grammatically correct sentences without syntax errors.
            Always use the correct spelling for each word.
            
            Valid SRT format example:
            1
            00:00:00,000 --> 00:00:02,500
            This is the first subtitle.
            
            2
            00:00:02,500 --> 00:00:05,000
            This is the second subtitle.
            """)
    ),
    # Spanish
    "es": Prompt.from_text(
        dedent("""\
            Eres un asistente útil que transcribe archivos de audio.
            El archivo de audio es {{audio_file}}.
            El idioma del archivo de audio es español.
            
            Devuelve SOLO la transcripción en formato SRT.
            Escribe oraciones gramaticalmente correctas sin errores de sintaxis.
            Siempre usa la ortografía correcta para cada palabra, incluyendo tildes y signos de puntuación.
            
            Ejemplo de formato SRT válido:
            1
            00:00:00,000 --> 00:00:02,500
            Este es el primer subtítulo.
            
            2
            00:00:02,500 --> 00:00:05,000
            Este es el segundo subtítulo.
            """)
    ),
    # Chinese (Mandarin)
    "zh": Prompt.from_text(
        dedent("""\
            你是一个有用的助手，负责转录音频文件。
            音频文件是 {{audio_file}}。
            音频文件的语言是中文。
            
            仅返回SRT格式的转录内容。
            书写语法正确的句子，没有语法错误。
            始终使用正确的汉字和标点符号。
            
            有效的SRT格式示例：
            1
            00:00:00,000 --> 00:00:02,500
            这是第一个字幕。
            
            2
            00:00:02,500 --> 00:00:05,000
            这是第二个字幕。
            """)
    ),
    # Hindi
    "hi": Prompt.from_text(
        dedent("""\
            आप एक सहायक हैं जो ऑडियो फ़ाइलों को ट्रांसक्राइब करते हैं।
            ऑडियो फ़ाइल {{audio_file}} है।
            ऑडियो फ़ाइल की भाषा हिंदी है।
            
            केवल SRT प्रारूप में ट्रांसक्रिप्शन लौटाएं।
            व्याकरणिक रूप से सही वाक्य लिखें, वाक्य रचना में त्रुटियों के बिना।
            हमेशा प्रत्येक शब्द की सही वर्तनी का उपयोग करें।
            
            मान्य SRT प्रारूप उदाहरण:
            1
            00:00:00,000 --> 00:00:02,500
            यह पहला उपशीर्षक है।
            
            2
            00:00:02,500 --> 00:00:05,000
            यह दूसरा उपशीर्षक है।
            """)
    ),
    # Arabic
    "ar": Prompt.from_text(
        dedent("""\
            أنت مساعد مفيد يقوم بنسخ الملفات الصوتية.
            الملف الصوتي هو {{audio_file}}.
            لغة الملف الصوتي هي العربية.
            
            قم بإرجاع النسخ بتنسيق SRT فقط.
            اكتب جملاً صحيحة نحوياً بدون أخطاء في بناء الجملة.
            استخدم دائماً التهجئة الصحيحة لكل كلمة.
            
            مثال على تنسيق SRT صالح:
            1
            00:00:00,000 --> 00:00:02,500
            هذه هي الترجمة الأولى.
            
            2
            00:00:02,500 --> 00:00:05,000
            هذه هي الترجمة الثانية.
            """)
    ),
    # Portuguese
    "pt": Prompt.from_text(
        dedent("""\
            Você é um assistente útil que transcreve arquivos de áudio.
            O arquivo de áudio é {{audio_file}}.
            O idioma do arquivo de áudio é português.
            
            Retorne APENAS a transcrição no formato SRT.
            Escreva frases gramaticalmente corretas sem erros de sintaxe.
            Sempre use a ortografia correta para cada palavra, incluindo acentos e pontuação.
            
            Exemplo de formato SRT válido:
            1
            00:00:00,000 --> 00:00:02,500
            Esta é a primeira legenda.
            
            2
            00:00:02,500 --> 00:00:05,000
            Esta é a segunda legenda.
            """)
    ),
    # Russian
    "ru": Prompt.from_text(
        dedent("""\
            Вы полезный помощник, который транскрибирует аудиофайлы.
            Аудиофайл: {{audio_file}}.
            Язык аудиофайла - русский.
            
            Верните ТОЛЬКО транскрипцию в формате SRT.
            Пишите грамматически правильные предложения без синтаксических ошибок.
            Всегда используйте правильное написание каждого слова.
            
            Пример правильного формата SRT:
            1
            00:00:00,000 --> 00:00:02,500
            Это первый субтитр.
            
            2
            00:00:02,500 --> 00:00:05,000
            Это второй субтитр.
            """)
    ),
    # Japanese
    "ja": Prompt.from_text(
        dedent("""\
            あなたは音声ファイルを文字起こしする便利なアシスタントです。
            音声ファイルは{{audio_file}}です。
            音声ファイルの言語は日本語です。
            
            SRT形式の文字起こしのみを返してください。
            文法的に正しい文章を書き、構文エラーがないようにしてください。
            常に各単語の正しい表記を使用してください。
            
            有効なSRT形式の例：
            1
            00:00:00,000 --> 00:00:02,500
            これは最初の字幕です。
            
            2
            00:00:02,500 --> 00:00:05,000
            これは二番目の字幕です。
            """)
    ),
    # German
    "de": Prompt.from_text(
        dedent("""\
            Sie sind ein hilfreicher Assistent, der Audiodateien transkribiert.
            Die Audiodatei ist {{audio_file}}.
            Die Sprache der Audiodatei ist Deutsch.
            
            Geben Sie NUR die Transkription im SRT-Format zurück.
            Schreiben Sie grammatikalisch korrekte Sätze ohne Syntaxfehler.
            Verwenden Sie immer die richtige Schreibweise für jedes Wort, einschließlich Umlaute.
            
            Beispiel für gültiges SRT-Format:
            1
            00:00:00,000 --> 00:00:02,500
            Dies ist der erste Untertitel.
            
            2
            00:00:02,500 --> 00:00:05,000
            Dies ist der zweite Untertitel.
            """)
    ),
    # French
    "fr": Prompt.from_text(
        dedent("""\
            Vous êtes un assistant utile qui transcrit des fichiers audio.
            Le fichier audio est {{audio_file}}.
            La langue du fichier audio est le français.
            
            Retournez UNIQUEMENT la transcription au format SRT.
            Écrivez des phrases grammaticalement correctes sans erreurs de syntaxe.
            Utilisez toujours l'orthographe correcte pour chaque mot, y compris les accents.
            
            Exemple de format SRT valide :
            1
            00:00:00,000 --> 00:00:02,500
            Ceci est le premier sous-titre.
            
            2
            00:00:02,500 --> 00:00:05,000
            Ceci est le deuxième sous-titre.
            """)
    ),
    # Italian
    "it": Prompt.from_text(
        dedent("""\
            Sei un assistente utile che trascrive file audio.
            Il file audio è {{audio_file}}.
            La lingua del file audio è l'italiano.
            
            Restituisci SOLO la trascrizione in formato SRT.
            Scrivi frasi grammaticalmente corrette senza errori di sintassi.
            Usa sempre l'ortografia corretta per ogni parola, inclusi gli accenti.
            
            Esempio di formato SRT valido:
            1
            00:00:00,000 --> 00:00:02,500
            Questo è il primo sottotitolo.
            
            2
            00:00:02,500 --> 00:00:05,000
            Questo è il secondo sottotitolo.
            """)
    ),
    # Korean
    "ko": Prompt.from_text(
        dedent("""\
            당신은 오디오 파일을 전사하는 유용한 도우미입니다.
            오디오 파일은 {{audio_file}}입니다.
            오디오 파일의 언어는 한국어입니다.
            
            SRT 형식의 전사본만 반환하세요.
            문법적으로 올바른 문장을 작성하고 구문 오류가 없어야 합니다.
            항상 각 단어의 올바른 철자를 사용하세요.
            
            유효한 SRT 형식 예시:
            1
            00:00:00,000 --> 00:00:02,500
            이것은 첫 번째 자막입니다.
            
            2
            00:00:02,500 --> 00:00:05,000
            이것은 두 번째 자막입니다.
            """)
    ),
    # Turkish
    "tr": Prompt.from_text(
        dedent("""\
            Ses dosyalarını yazıya döken yardımcı bir asistansınız.
            Ses dosyası {{audio_file}}.
            Ses dosyasının dili Türkçe.
            
            YALNIZCA SRT formatında transkripsiyonu döndürün.
            Dilbilgisi açısından doğru cümleler yazın, sözdizimi hataları olmadan.
            Her zaman her kelime için doğru yazımı kullanın.
            
            Geçerli SRT formatı örneği:
            1
            00:00:00,000 --> 00:00:02,500
            Bu ilk altyazıdır.
            
            2
            00:00:02,500 --> 00:00:05,000
            Bu ikinci altyazıdır.
            """)
    ),
    # Polish
    "pl": Prompt.from_text(
        dedent("""\
            Jesteś pomocnym asystentem, który transkrybuje pliki audio.
            Plik audio to {{audio_file}}.
            Język pliku audio to polski.
            
            Zwróć TYLKO transkrypcję w formacie SRT.
            Pisz zdania poprawne gramatycznie, bez błędów składniowych.
            Zawsze używaj poprawnej pisowni każdego słowa, w tym polskich znaków.
            
            Przykład prawidłowego formatu SRT:
            1
            00:00:00,000 --> 00:00:02,500
            To jest pierwszy napis.
            
            2
            00:00:02,500 --> 00:00:05,000
            To jest drugi napis.
            """)
    ),
    # Dutch
    "nl": Prompt.from_text(
        dedent("""\
            Je bent een behulpzame assistent die audiobestanden transcribeert.
            Het audiobestand is {{audio_file}}.
            De taal van het audiobestand is Nederlands.
            
            Retourneer ALLEEN de transcriptie in SRT-formaat.
            Schrijf grammaticaal correcte zinnen zonder syntaxisfouten.
            Gebruik altijd de juiste spelling voor elk woord.
            
            Voorbeeld van geldig SRT-formaat:
            1
            00:00:00,000 --> 00:00:02,500
            Dit is de eerste ondertitel.
            
            2
            00:00:02,500 --> 00:00:05,000
            Dit is de tweede ondertitel.
            """)
    ),
    # Indonesian
    "id": Prompt.from_text(
        dedent("""\
            Anda adalah asisten yang membantu mentranskripsikan file audio.
            File audio adalah {{audio_file}}.
            Bahasa file audio adalah Bahasa Indonesia.
            
            Kembalikan HANYA transkripsi dalam format SRT.
            Tulis kalimat yang benar secara tata bahasa tanpa kesalahan sintaks.
            Selalu gunakan ejaan yang benar untuk setiap kata.
            
            Contoh format SRT yang valid:
            1
            00:00:00,000 --> 00:00:02,500
            Ini adalah subtitle pertama.
            
            2
            00:00:02,500 --> 00:00:05,000
            Ini adalah subtitle kedua.
            """)
    ),
    # Vietnamese
    "vi": Prompt.from_text(
        dedent("""\
            Bạn là trợ lý hữu ích chuyển đổi file âm thanh thành văn bản.
            File âm thanh là {{audio_file}}.
            Ngôn ngữ của file âm thanh là tiếng Việt.
            
            Chỉ trả về bản phiên âm ở định dạng SRT.
            Viết câu đúng ngữ pháp, không có lỗi cú pháp.
            Luôn sử dụng chính tả đúng cho mỗi từ, bao gồm dấu thanh.
            
            Ví dụ định dạng SRT hợp lệ:
            1
            00:00:00,000 --> 00:00:02,500
            Đây là phụ đề đầu tiên.
            
            2
            00:00:02,500 --> 00:00:05,000
            Đây là phụ đề thứ hai.
            """)
    ),
    # Thai
    "th": Prompt.from_text(
        dedent("""\
            คุณเป็นผู้ช่วยที่ถอดความไฟล์เสียง
            ไฟล์เสียงคือ {{audio_file}}
            ภาษาของไฟล์เสียงคือภาษาไทย
            
            ส่งคืนเฉพาะการถอดความในรูปแบบ SRT เท่านั้น
            เขียนประโยคที่ถูกต้องตามหลักไวยากรณ์ ไม่มีข้อผิดพลาดทางไวยากรณ์
            ใช้การสะกดคำที่ถูกต้องสำหรับทุกคำเสมอ
            
            ตัวอย่างรูปแบบ SRT ที่ถูกต้อง:
            1
            00:00:00,000 --> 00:00:02,500
            นี่คือคำบรรยายแรก
            
            2
            00:00:02,500 --> 00:00:05,000
            นี่คือคำบรรยายที่สอง
            """)
    ),
    # Hebrew
    "he": Prompt.from_text(
        dedent("""\
            אתה עוזר שמתמלל קבצי אודיו.
            קובץ האודיו הוא {{audio_file}}.
            השפה של קובץ האודיו היא עברית.
            
            החזר רק את התמלול בפורמט SRT.
            כתוב משפטים נכונים מבחינה דקדוקית ללא שגיאות תחביר.
            השתמש תמיד באיות הנכון לכל מילה.
            
            דוגמה לפורמט SRT תקין:
            1
            00:00:00,000 --> 00:00:02,500
            זו הכתובית הראשונה.
            
            2
            00:00:02,500 --> 00:00:05,000
            זו הכתובית השנייה.
            """)
    ),
    # Swedish
    "sv": Prompt.from_text(
        dedent("""\
            Du är en hjälpsam assistent som transkriberar ljudfiler.
            Ljudfilen är {{audio_file}}.
            Språket i ljudfilen är svenska.
            
            Returnera ENDAST transkriptionen i SRT-format.
            Skriv grammatiskt korrekta meningar utan syntaxfel.
            Använd alltid korrekt stavning för varje ord, inklusive å, ä, ö.
            
            Exempel på giltigt SRT-format:
            1
            00:00:00,000 --> 00:00:02,500
            Detta är den första undertexten.
            
            2
            00:00:02,500 --> 00:00:05,000
            Detta är den andra undertexten.
            """)
    ),
    # Greek
    "el": Prompt.from_text(
        dedent("""\
            Είστε ένας χρήσιμος βοηθός που μεταγράφει αρχεία ήχου.
            Το αρχείο ήχου είναι {{audio_file}}.
            Η γλώσσα του αρχείου ήχου είναι ελληνικά.
            
            Επιστρέψτε ΜΟΝΟ τη μεταγραφή σε μορφή SRT.
            Γράψτε γραμματικά σωστές προτάσεις χωρίς συντακτικά λάθη.
            Χρησιμοποιείτε πάντα τη σωστή ορθογραφία για κάθε λέξη.
            
            Παράδειγμα έγκυρης μορφής SRT:
            1
            00:00:00,000 --> 00:00:02,500
            Αυτός είναι ο πρώτος υπότιτλος.
            
            2
            00:00:02,500 --> 00:00:05,000
            Αυτός είναι ο δεύτερος υπότιτλος.
            """)
    ),
    # Ukrainian
    "uk": Prompt.from_text(
        dedent("""\
            Ви корисний помічник, який транскрибує аудіофайли.
            Аудіофайл: {{audio_file}}.
            Мова аудіофайлу - українська.
            
            Поверніть ТІЛЬКИ транскрипцію у форматі SRT.
            Пишіть граматично правильні речення без синтаксичних помилок.
            Завжди використовуйте правильний правопис кожного слова.
            
            Приклад правильного формату SRT:
            1
            00:00:00,000 --> 00:00:02,500
            Це перший субтитр.
            
            2
            00:00:02,500 --> 00:00:05,000
            Це другий субтитр.
            """)
    ),
    # Czech
    "cs": Prompt.from_text(
        dedent("""\
            Jste užitečný asistent, který přepisuje zvukové soubory.
            Zvukový soubor je {{audio_file}}.
            Jazyk zvukového souboru je čeština.
            
            Vraťte POUZE přepis ve formátu SRT.
            Pište gramaticky správné věty bez syntaktických chyb.
            Vždy používejte správný pravopis každého slova včetně diakritiky.
            
            Příklad platného formátu SRT:
            1
            00:00:00,000 --> 00:00:02,500
            Toto je první titulek.
            
            2
            00:00:02,500 --> 00:00:05,000
            Toto je druhý titulek.
            """)
    ),
    # Romanian
    "ro": Prompt.from_text(
        dedent("""\
            Ești un asistent util care transcrie fișiere audio.
            Fișierul audio este {{audio_file}}.
            Limba fișierului audio este română.
            
            Returnează DOAR transcrierea în format SRT.
            Scrie propoziții corecte gramatical, fără erori de sintaxă.
            Folosește întotdeauna ortografia corectă pentru fiecare cuvânt, inclusiv diacritice.
            
            Exemplu de format SRT valid:
            1
            00:00:00,000 --> 00:00:02,500
            Aceasta este prima subtitrare.
            
            2
            00:00:02,500 --> 00:00:05,000
            Aceasta este a doua subtitrare.
            """)
    ),
    # Hungarian
    "hu": Prompt.from_text(
        dedent("""\
            Ön egy hasznos asszisztens, aki hangfájlokat ír át.
            A hangfájl: {{audio_file}}.
            A hangfájl nyelve magyar.
            
            CSAK az átírást adja vissza SRT formátumban.
            Nyelvtanilag helyes mondatokat írjon szintaktikai hibák nélkül.
            Mindig használja a helyes helyesírást minden szónál, beleértve az ékezeteket is.
            
            Érvényes SRT formátum példa:
            1
            00:00:00,000 --> 00:00:02,500
            Ez az első felirat.
            
            2
            00:00:02,500 --> 00:00:05,000
            Ez a második felirat.
            """)
    ),
    # Norwegian
    "no": Prompt.from_text(
        dedent("""\
            Du er en hjelpsom assistent som transkriberer lydfiler.
            Lydfilen er {{audio_file}}.
            Språket i lydfilen er norsk.
            
            Returner KUN transkripsjonen i SRT-format.
            Skriv grammatisk korrekte setninger uten syntaksfeil.
            Bruk alltid riktig stavemåte for hvert ord, inkludert æ, ø, å.
            
            Eksempel på gyldig SRT-format:
            1
            00:00:00,000 --> 00:00:02,500
            Dette er den første underteksten.
            
            2
            00:00:02,500 --> 00:00:05,000
            Dette er den andre underteksten.
            """)
    ),
    # Danish
    "da": Prompt.from_text(
        dedent("""\
            Du er en hjælpsom assistent, der transkriberer lydfiler.
            Lydfilen er {{audio_file}}.
            Sproget i lydfilen er dansk.
            
            Returner KUN transkriptionen i SRT-format.
            Skriv grammatisk korrekte sætninger uden syntaksfejl.
            Brug altid den korrekte stavemåde for hvert ord, inklusiv æ, ø, å.
            
            Eksempel på gyldigt SRT-format:
            1
            00:00:00,000 --> 00:00:02,500
            Dette er den første undertekst.
            
            2
            00:00:02,500 --> 00:00:05,000
            Dette er den anden undertekst.
            """)
    ),
    # Finnish
    "fi": Prompt.from_text(
        dedent("""\
            Olet hyödyllinen avustaja, joka litteroi äänitiedostoja.
            Äänitiedosto on {{audio_file}}.
            Äänitiedoston kieli on suomi.
            
            Palauta VAIN litterointi SRT-muodossa.
            Kirjoita kieliopillisesti oikeita lauseita ilman syntaksivirheitä.
            Käytä aina oikeaa kirjoitusasua jokaiselle sanalle, mukaan lukien ä ja ö.
            
            Esimerkki kelvollisesta SRT-muodosta:
            1
            00:00:00,000 --> 00:00:02,500
            Tämä on ensimmäinen tekstitys.
            
            2
            00:00:02,500 --> 00:00:05,000
            Tämä on toinen tekstitys.
            """)
    ),
}

logger = logging.getLogger(__name__)


class GoogleSpeechToTextProvider(SpeechToTextProvider):
    model: str

    def __init__(
        self,
        generation_provider: FailoverGenerationProvider | GoogleGenerationProvider,
        model: str = "gemini-1.5-flash",
    ) -> None:
        """
        Initializes the provider, ensuring that all nested providers are from Google.
        """
        self.model = model

        # Perform recursive validation.
        if not self._validate_google_providers_recursively(generation_provider):
            raise ValueError(
                "Invalid provider configuration. All providers, including those "
                + "within a nested FailoverGenerationProvider, must be instances of "
                + "GoogleGenerationProvider."
            )

        self._generation_provider = generation_provider

    def _validate_google_providers_recursively(
        self, provider: GenerationProviderType
    ) -> bool:
        """
        Recursively validates that a provider or a hierarchy of providers are all
        from Google.
        """
        # Base case: If the provider is a GoogleGenerationProvider, it's valid.
        if isinstance(provider, GoogleGenerationProvider):
            return True

        # Recursive step: If it's a Failover provider, check all its sub-providers.
        if isinstance(provider, FailoverGenerationProvider):
            # The 'all()' function ensures every sub-provider in the list is valid.
            # It will short-circuit and return False on the first invalid one found.
            return all(
                self._validate_google_providers_recursively(p)
                for p in provider.generation_providers
            )

        # If the provider is neither Google nor Failover, it's invalid in this context.
        return False

    async def transcribe_async(
        self, audio_file: str | Path, config: TranscriptionConfig | None = None
    ) -> AudioTranscription:
        """Gemini has the amazing 'ability' to transcribe audio files."""
        _config = config or TranscriptionConfig()
        language = _config.language or "en"

        path_audio_file = Path(audio_file)

        prompt = _LANGUAGE_TO_PROMPTS.get(language) or _LANGUAGE_TO_PROMPTS["en"]

        transcription = await self._generation_provider.generate_by_prompt_async(
            model=self.model,
            prompt=[
                FilePart(
                    data=path_audio_file.read_bytes(),
                    mime_type=ext2mime(path_audio_file.suffix),
                ),
                TextPart(text=prompt.compile(audio_file=audio_file)),
            ],
        )

        raw_transcription = transcription.text

        segments: Sequence[SentenceSegment] = self._parse_srt_to_segments(
            raw_transcription
        )

        subtitles = Subtitles(format="srt", subtitles=raw_transcription)

        prompt_tokens_used = transcription.usage.prompt_tokens
        completion_tokens_used = transcription.usage.completion_tokens

        ppmi = self._generation_provider.price_per_million_tokens_input(
            self.model, estimate_tokens=prompt_tokens_used
        )

        ppco = self._generation_provider.price_per_million_tokens_output(
            self.model, estimate_tokens=completion_tokens_used
        )

        cost: float = (
            prompt_tokens_used * ppmi + completion_tokens_used * ppco
        ) / 1_000_000

        # Get audio duration
        duration = await self._get_audio_duration(audio_file)

        return AudioTranscription(
            text=transcription.text,
            segments=segments,
            subtitles=subtitles,
            cost=cost,
            duration=duration,
        )

    async def _get_audio_duration(self, audio_file: str | Path) -> float:
        """Get the duration of an audio file using ffprobe."""
        try:
            # Use ffprobe to get audio duration
            process = await asyncio.create_subprocess_exec(
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(audio_file),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, _ = await process.communicate()

            if process.returncode != 0:
                # Fallback: try to estimate from file size (very rough estimate)
                file_path = Path(audio_file)
                if file_path.exists():
                    # Very rough estimate: assume 128kbps audio
                    file_size_bytes = file_path.stat().st_size
                    estimated_duration = file_size_bytes / (
                        128 * 1024 / 8
                    )  # 128kbps in bytes per second
                    return max(1.0, estimated_duration)  # At least 1 second
                return 1.0  # Default fallback

            # Parse ffprobe output
            probe_data = json.loads(stdout.decode())
            duration = float(probe_data["format"]["duration"])
            return duration

        except Exception:
            # Fallback: estimate from file size
            try:
                file_path = Path(audio_file)
                if file_path.exists():
                    file_size_bytes = file_path.stat().st_size
                    estimated_duration = file_size_bytes / (
                        128 * 1024 / 8
                    )  # 128kbps estimate
                    return max(1.0, estimated_duration)
            except Exception:
                pass
            return 1.0  # Final fallback

    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
        # Handle both comma and period as decimal separator
        timestamp = timestamp.replace(",", ".")

        parts = timestamp.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])

            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            raise ValueError(f"Invalid timestamp format: {timestamp}")

    def _fix_srt_format(self, srt_text: str) -> str:
        """Fix common SRT format issues."""
        lines = srt_text.strip().split("\n")
        fixed_lines: Sequence[str] = []

        i = 0
        segment_number = 1

        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue

            # Try to find a segment number (might be missing or incorrect)
            if lines[i].strip().isdigit():
                # Replace with correct segment number
                fixed_lines.append(str(segment_number))
                i += 1
            else:
                # Add missing segment number
                fixed_lines.append(str(segment_number))

            # Look for timestamp line
            if i < len(lines) and "-->" in lines[i]:
                fixed_lines.append(lines[i].strip())
                i += 1
            else:
                # Missing timestamp, skip this segment
                while i < len(lines) and lines[i].strip() and "-->" not in lines[i]:
                    i += 1
                continue

            # Collect text lines until empty line or next segment
            text_lines: MutableSequence[str] = []
            while (
                i < len(lines) and lines[i].strip() and not lines[i].strip().isdigit()
            ):
                if "-->" not in lines[i]:  # Make sure it's not a timestamp
                    text_lines.append(lines[i].strip())
                i += 1

            if text_lines:
                fixed_lines.extend(text_lines)
                fixed_lines.append("")  # Add empty line between segments
                segment_number += 1

        return "\n".join(fixed_lines)

    def _parse_srt_to_segments(self, raw_srt: str) -> Sequence[SentenceSegment]:
        """Parse SRT string into SentenceSegment objects."""
        segments = []

        srt_text = self._fix_srt_format(raw_srt)

        srt_pattern = re.compile(
            r"(\d+)\s*\n\s*"  # Segment number
            + r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*\n"  # Timestamps
            + r"((?:.*\n)*?)(?=\n\d+\s*\n|\n*$)",  # Text (lazy match until next segment or end)
            re.MULTILINE,
        )

        matches = srt_pattern.findall(srt_text)

        for match in matches:
            segment_id = int(match[0]) - 1  # Convert to 0-based index
            start_time = self._parse_timestamp(match[1])
            end_time = self._parse_timestamp(match[2])
            text = match[3].strip()

            # Skip empty segments
            if not text:
                continue

            segment = SentenceSegment(
                id=segment_id,
                sentence=text,
                start=start_time,
                end=end_time,
                no_speech_prob=-1.0,
            )
            segments.append(segment)

        # If no segments were parsed, try a more lenient approach
        if not segments:
            lines = srt_text.strip().split("\n")
            i = 0
            segment_id = 0

            while i < len(lines):
                # Skip empty lines
                if not lines[i].strip():
                    i += 1
                    continue

                # Skip segment numbers
                if lines[i].strip().isdigit():
                    i += 1

                # Look for timestamp line
                if i < len(lines) and "-->" in lines[i]:
                    try:
                        parts = lines[i].split("-->")
                        start_time = self._parse_timestamp(parts[0])
                        end_time = self._parse_timestamp(parts[1])
                        i += 1

                        # Collect text
                        text_lines: MutableSequence[str] = []
                        while (
                            i < len(lines)
                            and lines[i].strip()
                            and not lines[i].strip().isdigit()
                            and "-->" not in lines[i]
                        ):
                            text_lines.append(lines[i].strip())
                            i += 1

                        if text_lines:
                            segment = SentenceSegment(
                                id=segment_id,
                                sentence=" ".join(text_lines),
                                start=start_time,
                                end=end_time,
                                no_speech_prob=-1.0,
                            )
                            segments.append(segment)
                            segment_id += 1
                    except Exception:
                        i += 1
                else:
                    i += 1

        return segments
