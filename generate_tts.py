from gtts import gTTS
import os

texts = [
    "Please transfer five thousand rupees to my account immediately",
    "Hello I am calling from State Bank of India regarding your account",
    "Your account has been flagged for suspicious activity please verify",
    "I need to confirm your details before processing this transaction",
    "This is an urgent security alert regarding your bank account",
    "Please provide your OTP to complete the verification process",
    "Your credit card has been blocked please call us immediately",
    "We are offering you a special loan at very low interest rate",
    "Your account will be suspended if you do not verify now",
    "Hello this is your bank manager calling about your fixed deposit",
    "We need to update your KYC details to avoid account suspension",
    "Your transaction of ten thousand rupees is pending verification",
    "Please confirm your ATM PIN for security purposes",
    "This call is regarding your recent loan application status",
    "Your account shows unusual activity please verify immediately",
]

os.makedirs("tts_mic_fake", exist_ok=True)
for i, text in enumerate(texts):
    tts = gTTS(text=text, lang='en')
    tts.save(f"tts_mic_fake/gtts_fake_{i+1:02d}.mp3")
    print(f"Generated: gtts_fake_{i+1:02d}.mp3")

print("Done!")
