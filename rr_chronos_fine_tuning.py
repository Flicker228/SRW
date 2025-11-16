import os
import json
import numpy as np
import wfdb
from wfdb import processing
from sklearn.cluster import KMeans
import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, Trainer
from tqdm import tqdm
from chronos import ChronosPipeline
import joblib
import matplotlib.pyplot as plt


DATA_FOLDER = "014"
MODEL_NAME = "amazon/chronos-t5-small"
OUT_DIR = "chronos_rr_finetuned"
FS = 500
RR_RESAMPLE_LEN = 256
N_CLUSTERS = 1024
OFFSET = 2
PRED_RR = 2
BATCH = 4
EPOCHS = 40
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def detect_r_peaks(sig):
    return np.array(processing.xqrs_detect(sig=sig, fs=FS), dtype=int)

def extract_rr_intervals(sig, r_peaks):
    segs = []
    for i in range(len(r_peaks) - 1):
        a, b = r_peaks[i], r_peaks[i + 1]
        seg = sig[a:b].astype(np.float32)
        seg = (seg - seg.mean()) / (seg.std() + 1e-8)
        x_old = np.linspace(0, 1, num=len(seg))
        x_new = np.linspace(0, 1, num=RR_RESAMPLE_LEN)
        segs.append(np.interp(x_new, x_old, seg))
    return np.stack(segs, axis=0) if segs else np.empty((0, RR_RESAMPLE_LEN), dtype=np.float32)

def rr_collate_fn(batch):
    ids = [it['input_ids'] for it in batch]
    masks = [it['attention_mask'] for it in batch]
    labs = [it['labels'] for it in batch]
    input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

class RRTokenDataset(Dataset):
    def __init__(self, sequences, pred_len):
        self.samples = []
        for seq in sequences:
            inp = seq[:-pred_len]
            lab = seq[-pred_len:]
            self.samples.append({
                "input_ids": torch.tensor(inp).long(),
                "attention_mask": torch.ones(len(inp), dtype=torch.long),
                "labels": torch.tensor(lab).long()
            })
        print(f"Всего примеров для обучения: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def visualize_prediction(record_name=None, pred_rr=PRED_RR, out_dir=OUT_DIR):
    global hea_files, rr_per_record, token_sequences, record_signals, record_peaks

    if record_name is None:
        print("Укажите название записи")
        return

    pipeline = ChronosPipeline.from_pretrained(out_dir)
    model = getattr(pipeline, "inner_model", None) or pipeline.model
    model.to(DEVICE)
    model.eval()

    kmeans = joblib.load(os.path.join(out_dir, "kmeans_rr_model.pkl"))

    record_idx = next((i for i, h in enumerate(hea_files) if h[:-4] == record_name), -1)
    if record_idx == -1:
        print(f"Запись {record_name} не найдена")
        return

    s, e = rr_per_record[record_idx]
    seq_tokens = token_sequences[record_idx]
    original_signal = record_signals[record_idx]
    peaks = record_peaks[record_idx]

    context = seq_tokens[:-pred_rr]
    true_target = seq_tokens[-pred_rr:]

    with torch.no_grad():
        input_ids = torch.tensor([context], dtype=torch.long).to(DEVICE)
        generated = model.generate(input_ids=input_ids, max_length=len(context) + pred_rr, num_beams=1, do_sample=False, pad_token_id=0)
    pred_tokens = generated[0].cpu().numpy()[-pred_rr:]

    def token_to_shape(t):
        cid = t - OFFSET
        return kmeans.cluster_centers_[cid] if 0 <= cid < len(kmeans.cluster_centers_) else np.zeros(RR_RESAMPLE_LEN)

    context_end_peak = peaks[len(context)]
    target_peaks = peaks[len(context):len(context) + pred_rr + 1]
    start_viz = max(0, context_end_peak - 3 * FS)
    end_viz = min(len(original_signal), target_peaks[-1] + 2 * FS)
    time_axis = np.arange(start_viz, end_viz) / FS
    signal_viz = original_signal[start_viz:end_viz]

    plt.figure(figsize=(15, 10))
    plt.plot(time_axis, signal_viz, 'gray', alpha=0.8, linewidth=1, label='ЭКГ сигнал')
    viz_peaks = [p for p in peaks if start_viz <= p < end_viz]
    plt.scatter([p / FS for p in viz_peaks], original_signal[viz_peaks], color='red', s=40, zorder=5, label='R-пики')

    # контекстные (последние 3)
    for i in range(max(0, len(context) - 3), len(context)):
        a, b = peaks[i], peaks[i + 1]
        if a < start_viz or b > end_viz:
            continue
        plt.plot(np.arange(a, b) / FS, original_signal[a:b], 'blue', linewidth=2, label='Контекстные RR' if i == max(0, len(context) - 3) else "")

    # истинные целевые
    for i in range(len(target_peaks) - 1):
        a, b = target_peaks[i], target_peaks[i + 1]
        if a < start_viz or b > end_viz:
            continue
        plt.plot(np.arange(a, b) / FS, original_signal[a:b], 'green', linewidth=2, label='Истинные RR' if i == 0 else "")

    for i in range(pred_rr):
        if i >= len(target_peaks) - 1:
            break
        a, b = target_peaks[i], target_peaks[i + 1]
        if a < start_viz or b > end_viz:
            continue
        pred_shape = token_to_shape(pred_tokens[i])
        orig_len = b - a
        x_old = np.linspace(0, 1, len(pred_shape))
        x_new = np.linspace(0, 1, orig_len)
        pred_rs = np.interp(x_new, x_old, pred_shape)
        ctx = original_signal[max(0, a - 100):min(len(original_signal), a + 100)]
        if len(ctx) > 0:
            cm, cs = np.mean(ctx), np.std(ctx)
            pm, ps = pred_rs.mean(), pred_rs.std()
            pred_scaled = (pred_rs - pm) / ps * cs + cm if ps > 0 else pred_rs - pm + cm
        else:
            pred_scaled = pred_rs
        plt.plot(np.arange(a, b) / FS, pred_scaled, 'red', linestyle='--', linewidth=2, label='Предсказанные RR' if i == 0 else "")

    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.title(f"Предсказание RR-интервалов: {record_name}")
    plt.legend()
    plt.grid(alpha=0.3)
    for p in viz_peaks:
        plt.axvline(x=p / FS, color='red', alpha=0.3, linestyle=':')
    plt.tight_layout()
    plt.show()


hea_files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith(".hea")])
all_rr = []
rr_per_record = []
record_signals = []
record_peaks = []

print("Извлечение RR-интервалов из записей...")
for hea in tqdm(hea_files):
    name = hea[:-4]
    rec = wfdb.rdrecord(os.path.join(DATA_FOLDER, name))
    sig = rec.p_signal[:, 1].astype(np.float32)
    sig_n = (sig - sig.mean()) / (sig.std() + 1e-8)
    peaks = detect_r_peaks(sig_n)
    rr_intervals = extract_rr_intervals(sig_n, peaks)
    start = len(all_rr)
    if rr_intervals.shape[0] > 0:
        all_rr.extend(list(rr_intervals))
        end = len(all_rr)
    else:
        end = start
    rr_per_record.append((start, end))
    record_signals.append(sig)
    record_peaks.append(peaks)

all_rr = np.stack(all_rr, axis=0).astype(np.float32)
print("Всего RR-интервалов:", all_rr.shape[0], "длина интервала:", all_rr.shape[1])

pipeline = ChronosPipeline.from_pretrained(MODEL_NAME)
model = getattr(pipeline, "inner_model", None) or pipeline.model
vocab_size = getattr(model.config, "vocab_size", None) or 32000
max_allowed = max(4, vocab_size - 10 - OFFSET)
n_clusters = min(N_CLUSTERS, max_allowed)
print(f"Используем n_clusters={n_clusters} (vocab_size={vocab_size})")

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(all_rr)
labels_all = kmeans.labels_

token_sequences = []
for s, e in rr_per_record:
    if e <= s:
        token_sequences.append(np.zeros((0,), dtype=np.int64))
    else:
        token_sequences.append(labels_all[s:e].astype(np.int64) + OFFSET)

dataset = RRTokenDataset(token_sequences, PRED_RR)

training_args = TrainingArguments(
    output_dir=os.path.join(OUT_DIR, "ckpt"),
    per_device_train_batch_size=BATCH,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    weight_decay=0.0,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, data_collator=rr_collate_fn)
trainer.train()

os.makedirs(OUT_DIR, exist_ok=True)
joblib.dump(kmeans, os.path.join(OUT_DIR, "kmeans_rr_model.pkl"))
np.save(os.path.join(OUT_DIR, "kmeans_centers.npy"), kmeans.cluster_centers_)
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump({"n_clusters": int(n_clusters), "rr_resample_len": int(RR_RESAMPLE_LEN), "pred_rr": int(PRED_RR), "offset": int(OFFSET), "fs": int(FS)}, f, indent=2)

model.save_pretrained(OUT_DIR)
print(f"Модель сохранена в {OUT_DIR}")

visualize_prediction(record_name="JS00415", pred_rr=2)
