import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import Optional

import discord
import keyring

logger = logging.getLogger(__name__)

DISCORD_CHANNEL = "news3"


# ##################################################################
# extract title from xml
# parse xml content and extract the title element
def extract_title_from_xml(xml_content: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_content)
        title_elem = root.find("title")
        if title_elem is not None and title_elem.text:
            return title_elem.text
        return None
    except ET.ParseError as err:
        logger.error("Failed to parse XML: %s", err)
        return None


# ##################################################################
# extract summary from xml
# parse xml content and extract the summary element
def extract_summary_from_xml(xml_content: str) -> Optional[str]:
    try:
        root = ET.fromstring(xml_content)
        summary_elem = root.find("summary")
        if summary_elem is not None and summary_elem.text:
            return summary_elem.text
        return None
    except ET.ParseError as err:
        logger.error("Failed to parse XML: %s", err)
        return None


# ##################################################################
# format news message
# create a clean discord message for a news item
def format_news_message(title: str, link: str, score: float, feed_name: str, summary: Optional[str] = None) -> str:
    lines = []

    # header with score and source
    lines.append(f"**{score:.1f}** · {feed_name}")
    lines.append("")

    # title as bold
    lines.append(f"**{title}**")

    # summary if available (truncate if too long)
    if summary:
        if len(summary) > 200:
            summary = summary[:197] + "..."
        lines.append(summary)

    lines.append("")

    # link at the end (Discord will auto-embed)
    lines.append(link)

    return "\n".join(lines)


# ##################################################################
# send to discord
# send a message to a channel using discord.py directly
def send_to_discord(message: str, channel_name: str = DISCORD_CHANNEL) -> bool:
    token = keyring.get_password("discord_events", "token")
    if not token:
        logger.error("Discord token not found in keyring (discord_events/token)")
        return False

    result = {"success": False}

    async def send():
        intents = discord.Intents.default()
        intents.guilds = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_ready():
            for channel in client.get_all_channels():
                if isinstance(channel, discord.TextChannel) and channel.name == channel_name:
                    await channel.send(message)
                    logger.info("Sent message to #%s", channel_name)
                    result["success"] = True
                    break
            else:
                logger.error("Channel #%s not found", channel_name)

            await client.close()

        await client.start(token)

    try:
        asyncio.run(send())
        return result["success"]
    except Exception as err:
        logger.error("Discord error: %s", err)
        return False


# ##################################################################
# send news item
# send a scored news entry to discord with nice formatting
def send_news_item(title: str, link: str, score: float, feed_name: str, summary: Optional[str] = None) -> bool:
    message = format_news_message(title, link, score, feed_name, summary)
    return send_to_discord(message)
