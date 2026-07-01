"""손글씨 숫자 인식기 (MNIST CNN + Tkinter 캔버스)

처음 실행하면 MNIST를 내려받아 CNN을 학습한 뒤 model.pth로 저장합니다.
이후 실행은 캐시된 가중치를 바로 로드합니다.
캔버스에 마우스로 숫자를 그리고 "인식" 버튼을 누르세요.
"""
from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "mnist_cnn.pth"
DATA_DIR = HERE / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def train_model(epochs: int = 2) -> CNN:
    print(f"[학습] MNIST 다운로드 + {epochs} 에폭 학습 시작 (device={DEVICE})", flush=True)
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST(str(DATA_DIR), train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(str(DATA_DIR), train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    model = CNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            if batch_idx % 100 == 0:
                print(f"  epoch {epoch} batch {batch_idx}/{len(train_loader)} loss={loss.item():.4f}", flush=True)
        # eval
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        print(f"  epoch {epoch} test acc = {correct / total:.4f}", flush=True)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"[학습] 가중치 저장: {MODEL_PATH}", flush=True)
    return model


def load_or_train() -> CNN:
    model = CNN().to(DEVICE)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"[로드] 캐시된 가중치 사용: {MODEL_PATH}", flush=True)
        return model
    return train_model()


def preprocess(img: Image.Image) -> torch.Tensor:
    """캔버스 PIL 이미지를 MNIST 형식 텐서로 변환.

    1) 그레이스케일 + 반전(MNIST는 검은 배경/흰 글씨)
    2) 글씨 bbox로 크롭 → 20x20에 맞게 비율 보존 리사이즈
    3) 28x28 중앙에 배치 (질량 중심 보정)
    """
    g = img.convert("L")
    inv = ImageOps.invert(g)
    arr = np.array(inv)
    ys, xs = np.where(arr > 20)
    if len(xs) == 0:
        return torch.zeros(1, 1, 28, 28, device=DEVICE)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    cropped = inv.crop((x0, y0, x1, y1))

    w, h = cropped.size
    if w > h:
        new_w = 20
        new_h = max(1, int(round(h * 20 / w)))
    else:
        new_h = 20
        new_w = max(1, int(round(w * 20 / h)))
    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(resized, ((28 - new_w) // 2, (28 - new_h) // 2))

    a = np.array(canvas, dtype=np.float32)
    total = a.sum()
    if total > 0:
        cy = (a.sum(axis=1) * np.arange(28)).sum() / total
        cx = (a.sum(axis=0) * np.arange(28)).sum() / total
        shift_x = int(round(14 - cx))
        shift_y = int(round(14 - cy))
        canvas = ImageOps.expand(canvas, border=8, fill=0)
        canvas = canvas.transform(
            (28, 28), Image.AFFINE, (1, 0, 8 - shift_x, 0, 1, 8 - shift_y), fillcolor=0
        )

    a = np.array(canvas, dtype=np.float32) / 255.0
    a = (a - 0.1307) / 0.3081
    return torch.from_numpy(a).unsqueeze(0).unsqueeze(0).to(DEVICE)


class App:
    CANVAS = 280  # 10x scale of 28

    def __init__(self, model: CNN) -> None:
        self.model = model
        self.root = tk.Tk()
        self.root.title("손글씨 숫자 인식기")
        self.root.resizable(False, False)

        self.image = Image.new("L", (self.CANVAS, self.CANVAS), "white")
        self.draw = ImageDraw.Draw(self.image)

        self.canvas = tk.Canvas(self.root, width=self.CANVAS, height=self.CANVAS, bg="white", cursor="pencil")
        self.canvas.grid(row=0, column=0, columnspan=3, padx=8, pady=8)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<ButtonRelease-1>", lambda _e: self.predict())
        self.last_xy: tuple[int, int] | None = None
        self.canvas.bind("<ButtonPress-1>", self.on_press)

        tk.Button(self.root, text="인식", width=10, command=self.predict).grid(row=1, column=0, pady=4)
        tk.Button(self.root, text="지우기", width=10, command=self.clear).grid(row=1, column=1, pady=4)
        tk.Button(self.root, text="종료", width=10, command=self.root.destroy).grid(row=1, column=2, pady=4)

        self.result_var = tk.StringVar(value="여기에 숫자를 그리세요")
        tk.Label(self.root, textvariable=self.result_var, font=("Segoe UI", 18)).grid(
            row=2, column=0, columnspan=3, pady=(4, 2)
        )
        self.probs_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.probs_var, font=("Consolas", 10), justify="left").grid(
            row=3, column=0, columnspan=3, pady=(0, 8)
        )

    def on_press(self, e: tk.Event) -> None:
        self.last_xy = (e.x, e.y)

    def on_draw(self, e: tk.Event) -> None:
        r = 12
        if self.last_xy is not None:
            x0, y0 = self.last_xy
            self.canvas.create_line(x0, y0, e.x, e.y, fill="black", width=r * 2, capstyle=tk.ROUND, smooth=True)
            self.draw.line([x0, y0, e.x, e.y], fill="black", width=r * 2)
        self.canvas.create_oval(e.x - r, e.y - r, e.x + r, e.y + r, fill="black", outline="black")
        self.draw.ellipse([e.x - r, e.y - r, e.x + r, e.y + r], fill="black")
        self.last_xy = (e.x, e.y)

    def clear(self) -> None:
        self.canvas.delete("all")
        self.image = Image.new("L", (self.CANVAS, self.CANVAS), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_var.set("여기에 숫자를 그리세요")
        self.probs_var.set("")
        self.last_xy = None

    def predict(self) -> None:
        x = preprocess(self.image)
        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1).cpu().numpy().ravel()
        top = int(probs.argmax())
        self.result_var.set(f"예측: {top}   (신뢰도 {probs[top] * 100:.1f}%)")
        bars = "\n".join(f"  {d}: {'█' * int(round(probs[d] * 20)):<20s} {probs[d] * 100:5.1f}%" for d in range(10))
        self.probs_var.set(bars)

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    model = load_or_train()
    App(model).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
