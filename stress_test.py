#!/usr/bin/env python3
"""
PaddleOCR API 压力测试
用法: python stress_test.py [并发数] [请求数]
默认: 并发2, 请求10
"""
import asyncio
import sys
import time
import statistics

import aiohttp
from pathlib import Path

TEST_IMG = Path(__file__).parent / "models" / "test.png"
API_URL = "http://localhost:8899/ocr"

CONCURRENT = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 10

times = []
errors = []


async def do_ocr(session, i):
    """单次 OCR 请求"""
    t0 = time.time()
    try:
        with open(TEST_IMG, "rb") as f:
            form = aiohttp.FormData()
            form.add_field("file", f, filename="test.png", content_type="image/png")
            async with session.post(API_URL, data=form, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                data = await resp.json()
                t1 = time.time()
                elapsed = t1 - t0
                if "error" in data:
                    errors.append((i, data["error"]))
                    print(f"  [{i:>3}] ❌ {elapsed:.2f}s error: {data['error']}")
                else:
                    times.append(elapsed)
                    count = data.get("count", 0)
                    print(f"  [{i:>3}] ✅ {elapsed:.2f}s | {count} texts | {data.get('texts', [])[:3]}...")
    except Exception as e:
        t1 = time.time()
        errors.append((i, str(e)))
        print(f"  [{i:>3}] ❌ {time.time()-t0:.2f}s exception: {e}")


async def main():
    print(f"\n{'='*60}")
    print(f"  PaddleOCR API 压力测试")
    print(f"  并发={CONCURRENT}  请求={TOTAL}  接口={API_URL}")
    print(f"{'='*60}\n")

    t_start = time.time()

    async with aiohttp.ClientSession() as session:
        tasks = []
        sem = asyncio.Semaphore(CONCURRENT)

        async def bounded(i):
            async with sem:
                await do_ocr(session, i)

        for i in range(TOTAL):
            tasks.append(asyncio.create_task(bounded(i + 1)))

        await asyncio.gather(*tasks)

    t_total = time.time() - t_start

    print(f"\n{'='*60}")
    print(f"  📊 结果统计")
    print(f"{'='*60}")
    print(f"  总请求:       {TOTAL}")
    print(f"  成功:         {len(times)}")
    print(f"  失败:         {len(errors)}")
    print(f"  总耗时:       {t_total:.1f}s")
    print(f"  QPS (含排队):  {TOTAL/t_total:.2f}")
    if times:
        print(f"  平均响应:     {statistics.mean(times):.2f}s")
        print(f"  中位数:       {statistics.median(times):.2f}s")
        print(f"  最快:         {min(times):.2f}s")
        print(f"  最慢:         {max(times):.2f}s")
        if len(times) > 1:
            print(f"  标准差:       {statistics.stdev(times):.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if not TEST_IMG.exists():
        print(f"❌ 测试图片不存在: {TEST_IMG}")
        sys.exit(1)
    asyncio.run(main())
