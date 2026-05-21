import requests, os
from dotenv import load_dotenv
load_dotenv("/home/ubuntu/poly-model/.env")
wallet = os.getenv("FUNDER")
all_pos = []
offset = 0
while True:
    r = requests.get(
        f"https://data-api.polymarket.com/positions?user={wallet}&sizeThreshold=0.01&limit=100&offset={offset}",
        timeout=15
    )
    page = r.json()
    if not page:
        break
    all_pos.extend(page)
    if len(page) < 100:
        break
    offset += 100

redeemable = [p for p in all_pos if p.get("redeemable") and p.get("curPrice", 0) >= 0.97]
pending_uma = [p for p in all_pos if not p.get("redeemable") and p.get("curPrice", 0) >= 0.97]
r_val = sum(p.get("size", 0) for p in redeemable)
u_val = sum(p.get("size", 0) for p in pending_uma)
print(f"Total positions: {len(all_pos)}")
print(f"Redeemable now:  {len(redeemable)}  value=${r_val:.2f}")
print(f"UMA pending:     {len(pending_uma)}  value=${u_val:.2f}")
if pending_uma:
    print("\nUMA pending breakdown:")
    for p in sorted(pending_uma, key=lambda x: -x.get("size", 0)):
        title = p.get("title", p.get("market", ""))
        print(f"  ${p.get('size', 0):.2f}  curPrice={p.get('curPrice', 0):.3f}  {title[:55]}")
