import edge_tts
import asyncio

async def main():
    tts = edge_tts.Communicate("你好，这是一个测试。", "zh-CN-XiaoxiaoNeural")
    await tts.save("output.mp3")

asyncio.run(main())
