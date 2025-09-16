import os
import mmcv
import cv2
import numpy as np
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from mmcls.apis.inference import init_model, inference_model
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from retinaface import RetinaFace


# --- Токены ---
TELEGRAM_TOKEN = "8111604463:AAGqn6QXgT6ssMyZNQaZt_LiiQ2ft5MRp14"
YOUTUBE_API_KEY = "AIzaSyAm0DAHsqepk5o4U3dc1mhFyBB8gpwMVJk"

max_results = 50

# --- Инициализация модели APViT ---
MODEL = init_model(
    config='configs/apvit/RAF.py',
    checkpoint='weights/APViT_RAF-3eeecf7d.pth',
    device='cpu'
)

# --- Инициализация YouTube API ---
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Пришли мне фото, и я предскажу эмоцию на нём 📷"
    )

def search_playlist(emotion, max_results=1):
    try:
        emotion_to_search = {
            "Happiness": "Happy",
            "Sadness": "Sad",
            "Anger": "Angry",
            "Surprise": "Surprise",
            "Disgust": "Disturbing",
            "Fear": "Scary",
            "Neutral": "Chill"
        }
        request = youtube.search().list(
            part="snippet",
            q=f"{emotion_to_search.get(emotion)} Music Playlist",
            type="playlist",
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        items = response.get("items", [])
        if not items:
            return "Плейлист не найден"
        playlist_id = items[0]["id"]["playlistId"]
        return f"https://www.youtube.com/playlist?list={playlist_id}"
    except HttpError as e:
        return f"Ошибка при поиске плейлиста: {e}"

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    file_path = f"{photo.file_unique_id}.jpg"
    await file.download_to_drive(file_path)

    save_path = "aligned_face.jpg"
    if align_and_crop_face(file_path, save_path) is None:
        message = "Лицо не обнаружено"
        await update.message.reply_text(message)
    else:
        img = mmcv.imread(save_path)
        result = inference_model(MODEL, img)
        os.remove(file_path)

        pred_class = result['pred_class']
        pred_score = result['pred_score']

        # Динамически ищем плейлист по эмоции
        playlist_url = search_playlist(pred_class)

        message = (
            f"Эмоция: {pred_class}\n"
            f"Уверенность: {pred_score:.2%}\n"
            f"Рекомендованный плейлист: {playlist_url}"
        )

        await update.message.reply_text(message)

def align_and_crop_face(image_path, save_path="aligned_face.jpg", target_size=224):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Ошибка: не удалось открыть файл {image_path}")
        return None

    height, width = img.shape[:2]
    percent = 0.4
    border_h = int(height * percent)
    border_w = int(width * percent)

    img = cv2.copyMakeBorder(
        img,
        top=border_h,
        bottom=border_h,
        left=border_w,
        right=border_w,
        borderType=cv2.BORDER_CONSTANT,
        value=(255, 255, 255)
    )

    border_path = "border.jpg"
    cv2.imwrite(border_path, img)
    faces = RetinaFace.detect_faces(border_path)
    if not faces:
        print("Лицо не найдено!")
        return None

    face = list(faces.values())[0]
    landmarks = face["landmarks"]
    left_eye, right_eye = landmarks["left_eye"], landmarks["right_eye"]

    if left_eye[0] > right_eye[0]:
        left_eye, right_eye = right_eye, left_eye

    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    center = (float((left_eye[0] + right_eye[0]) / 2),
              float((left_eye[1] + right_eye[1]) / 2))

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    x1, y1, x2, y2 = face["facial_area"]
    w, h = x2 - x1, y2 - y1
    size = int(max(w, h))

    # центр лица
    cx, cy = x1 + w//2, y1 + h//2

    # квадратный кроп с сохранением пропорций
    x1_crop = max(0, cx - size//2)
    y1_crop = max(0, cy - size//2)
    x2_crop = min(rotated.shape[1], cx + size//2)
    y2_crop = min(rotated.shape[0], cy + size//2)

    face_crop = rotated[y1_crop:y2_crop, x1_crop:x2_crop]

    # resize с сохранением пропорций + padding
    h_crop, w_crop = face_crop.shape[:2]
    scale = target_size / max(h_crop, w_crop)
    new_w, new_h = int(w_crop * scale), int(h_crop * scale)
    resized = cv2.resize(face_crop, (new_w, new_h))

    # создаем квадрат 224x224
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    final = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.imwrite(save_path, final)
    print(f"Сохранено: {save_path}")
    return final

def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
