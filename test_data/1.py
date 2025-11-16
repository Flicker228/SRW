import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosPipeline
import wfdb

save_dir = "/kaggle/working/chronos_finetuned_pipeline"  # путь к вашей fine-tuned модели
pipeline = ChronosPipeline.from_pretrained(save_dir)


def predict_ecg(pipeline, context_signal, prediction_length=250):
    ctx = context_signal.astype(np.float32)
    ctx = (ctx - ctx.mean()) / (ctx.std() + 1e-8)
    ctx_tensor = torch.from_numpy(ctx)

    forecast = pipeline.predict([ctx_tensor], prediction_length=prediction_length, num_samples=1)
    return forecast[0].squeeze()


record_path = "/kaggle/input/12-lead-ecg/250/JS24400"
record = wfdb.rdrecord(record_path)
ecg_signal = record.p_signal[:, 1]

context_length = 2000
prediction_length = 250

context_signal = ecg_signal[:context_length]

predicted_segment = predict_ecg(pipeline, context_signal, prediction_length=prediction_length)

plt.figure(figsize=(12, 5))
plt.plot(np.arange(context_length), context_signal, label='Контекст (настоящий сигнал)', color='blue')
plt.plot(
    np.arange(context_length, context_length + prediction_length),
    ecg_signal[context_length:context_length + prediction_length],
    label='Реальный будущий сигнал', color='green'
)
plt.plot(
    np.arange(context_length, context_length + prediction_length),
    predicted_segment,
    label='Предсказанный сигнал', color='red', linestyle='--'
)

plt.xlabel('Время (сэмплы)')
plt.ylabel('Амплитуда ECG')
plt.title('Прогноз ECG с ChronosPipeline')
plt.legend()
plt.show()
