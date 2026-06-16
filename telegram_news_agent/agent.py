from __future__ import annotations

import argparse
import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from telethon import TelegramClient


URL_PATTERN = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class NewsItem:
    channel: str
    message_id: int
    date: datetime
    text: str


@dataclass(frozen=True)
class NewsCluster:
    items: list[NewsItem]
    summary: str


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def normalize_text(text: str) -> str:
    without_links = URL_PATTERN.sub("", text)
    return SPACE_PATTERN.sub(" ", without_links).strip()


async def fetch_news(config: dict[str, Any]) -> list[NewsItem]:
    telegram_config = config["telegram"]
    news_config = config["news"]
    min_text_length = int(news_config.get("min_text_length", 80))
    since = datetime.now(timezone.utc) - timedelta(hours=int(news_config.get("lookback_hours", 24)))

    client = TelegramClient(
        telegram_config.get("session_name", "news_agent"),
        int(telegram_config["api_id"]),
        telegram_config["api_hash"],
    )

    items: list[NewsItem] = []
    async with client:
        for channel in telegram_config["source_channels"]:
            async for message in client.iter_messages(
                channel,
                limit=int(news_config.get("max_messages_per_channel", 100)),
            ):
                if message.date < since or not message.message:
                    continue

                text = normalize_text(message.message)
                if len(text) < min_text_length:
                    continue

                items.append(
                    NewsItem(
                        channel=str(channel),
                        message_id=int(message.id),
                        date=message.date,
                        text=text,
                    )
                )

    return sorted(items, key=lambda item: item.date)


async def build_embeddings(client: AsyncOpenAI, model: str, texts: list[str]) -> np.ndarray:
    response = await client.embeddings.create(model=model, input=texts)
    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def cluster_news(items: list[NewsItem], embeddings: np.ndarray, threshold: float) -> list[list[NewsItem]]:
    clusters: list[list[NewsItem]] = []
    cluster_vectors: list[list[np.ndarray]] = []
    centroids: list[np.ndarray] = []

    for item, vector in zip(items, embeddings, strict=True):
        best_index = -1
        best_score = 0.0

        for index, centroid in enumerate(centroids):
            score = cosine_similarity(vector, centroid)
            if score > best_score:
                best_index = index
                best_score = score

        if best_index >= 0 and best_score >= threshold:
            clusters[best_index].append(item)
            cluster_vectors[best_index].append(vector)
            centroids[best_index] = np.array(cluster_vectors[best_index]).mean(axis=0)
        else:
            clusters.append([item])
            cluster_vectors.append([vector])
            centroids.append(vector)

    return clusters


async def summarize_cluster(client: AsyncOpenAI, model: str, cluster: list[NewsItem], language: str) -> str:
    sources = "\n\n".join(
        f"Источник: {item.channel}, дата: {item.date.isoformat()}\n{item.text}"
        for item in cluster
    )
    prompt = (
        "Собери одну цельную новость из похожих Telegram-постов. "
        "Исключи повторы, сохрани важные факты, числа, даты и разные детали из источников. "
        "Если источники противоречат друг другу, явно укажи это. "
        f"Ответь на языке: {language}.\n\n{sources}"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Ты редактор новостного дайджеста. Пиши кратко, точно и нейтрально."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


async def build_digest(config: dict[str, Any], items: list[NewsItem]) -> str:
    if not items:
        return "За выбранный период новых сообщений для дайджеста не найдено."

    openai_config = config["openai"]
    news_config = config["news"]
    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    embeddings = await build_embeddings(client, openai_config["embedding_model"], [item.text for item in items])
    raw_clusters = cluster_news(items, embeddings, float(news_config.get("similarity_threshold", 0.82)))

    clusters: list[NewsCluster] = []
    for raw_cluster in raw_clusters:
        summary = await summarize_cluster(
            client,
            openai_config["model"],
            raw_cluster,
            news_config.get("language", "ru"),
        )
        clusters.append(NewsCluster(items=raw_cluster, summary=summary))

    digest_parts = ["🗞️ Дайджест новостей"]
    for index, cluster in enumerate(clusters, start=1):
        sources = ", ".join(sorted({item.channel for item in cluster.items}))
        digest_parts.append(f"\n{index}. {cluster.summary}\nИсточники: {sources}")

    return "\n".join(digest_parts)


async def send_digest(config: dict[str, Any], digest: str) -> None:
    telegram_config = config["telegram"]
    client = TelegramClient(
        telegram_config.get("session_name", "news_agent"),
        int(telegram_config["api_id"]),
        telegram_config["api_hash"],
    )

    async with client:
        await client.send_message(telegram_config["target_chat"], digest)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram semantic news digest agent")
    parser.add_argument("--config", type=Path, default=Path("telegram_news_agent/config.yaml"))
    args = parser.parse_args()

    load_dotenv()
    config = load_config(args.config)
    news_items = await fetch_news(config)
    digest = await build_digest(config, news_items)
    await send_digest(config, digest)


if __name__ == "__main__":
    asyncio.run(main())
