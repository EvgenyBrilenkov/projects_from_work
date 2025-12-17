from datetime import datetime, timezone, timedelta
from pathlib import Path
from ultralytics import YOLO
import cv2

# ===== CONFIG =====
MODEL_PATH = "/wrk/main/yolo/runs/segment/best_results/yolo11-l/train/weights/best.pt" # Сделал директорию best_results, куда положил лучшую модель
PROJECT_ROOT = Path("/wrk/main/production_test")

# ===== NEW DIRS =====
TODAY_STR = datetime.now().strftime("%d_%m_%Y")
TODAY_DIR = PROJECT_ROOT / TODAY_STR
INCOMING_DIR = PROJECT_ROOT / "В_процессе"

TODAY_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(exist_ok=True)

print(f"Сегодня: {TODAY_STR}")
print(f"Рабочая директория: {TODAY_DIR}")

# ===== STARTING =====
model = YOLO(MODEL_PATH)

counter = 0
bad = 0
good = 0
defect_ids = []

def has_defect(results):
    return len(results[0].boxes) > 0

def process_part(part_id, first_img, second_img):
    global counter, bad, good
    try:
        part_dir = TODAY_DIR / str(part_id)
        orig_dir = part_dir / "Оригинал"
        pred_dir = part_dir / "Предсказания"
        orig_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)

        res_1 = model(first_img, device=0, verbose=False)[0]
        res_2 = model(second_img, device=0, verbose=False)[0]

        res_1.save(filename=str(pred_dir / "Результат_1.png"))
        res_2.save(filename=str(pred_dir / "Результат_2.png"))

        cv2.imwrite(str(orig_dir / "Оригинал_1.png"), cv2.imread(str(first_img)))
        cv2.imwrite(str(orig_dir / "Оригинал_2.png"), cv2.imread(str(second_img)))

        defect = has_defect([res_1]) + has_defect([res_2])

        if defect:
            with open(part_dir / "Вывод.txt", "w") as f:
                f.write("ДЕФЕКТ")
            
            bad += 1
            defect_ids.append(str(counter))

            with open(TODAY_DIR / "Бракованные_ID.txt", "w") as f:
                f.write('\n'.join(defect_ids))
        else:
            with open(part_dir / "Вывод.txt", "w") as f:
                f.write("НОРМА")

            good += 1

        counter += 1
        return defect
    except Exception as e:
        with open(part_dir / "Ошибка.txt", "w") as f:
            f.write(f"Возникла ошибка:\n{e}")

while True:
    files = [p for p in INCOMING_DIR.iterdir()]
    
    if len(files) > 2:
        with open(INCOMING_DIR / "Ошибка.txt", "w") as f:
                f.write(f"Количество фотографий в потоковой папке 'В_процессе' больше 2-х. Дата ошибки: {TODAY_STR}")
        print(f'Ошибка количества файлов. Дополнительные данные об ошибке: {INCOMING_DIR / "Ошибка.txt"}')
        break
    
    if len(files) == 2:
        process_part(counter, files[0], files[1])
        files[1].unlink()
        files[0].unlink()
        
        time_now = datetime.now(timezone(timedelta(hours=3))).strftime('%H:%M:%S %d-%m-%Y')
        print("\n")
        print(f"Последнее время отработки: {time_now}")
        print(f"Количество нормальных деталей: {good}")
        print(f"Количество дефектных деталей: {bad}")
        print(f"Процент плохих деталей за сегодня: {round((bad/(good+bad))*100, 1)}%")
        print(f"Общее количество обработанных деталей за сегодня: {bad+good}")

        with open(TODAY_DIR / "Сводка.txt", "w") as f:
            f.write(f"Сейчас: {time_now}\nКоличество нормальных деталей: {good}\nКоличество дефектных деталей: {bad}\nПроцент плохих деталей за сегодня: {round((bad/(good+bad))*100, 1)}%\nОбщее количество обработанных деталей за сегодня: {bad+good}")