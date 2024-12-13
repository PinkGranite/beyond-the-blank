import assemblyai as aai


class Whisper:
    def __init__(self):
        aai.settings.api_key = "b87ebe668b104115aea6d2f5ce11fd95"
        

    def transcribe(self):
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe('./output.mp3')

        if transcript.status == aai.TranscriptStatus.error:
            print(transcript.error)
        else:
            print(transcript.text)
            return transcript.text