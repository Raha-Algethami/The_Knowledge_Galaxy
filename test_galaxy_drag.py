import asyncio
from playwright.async_api import async_playwright

async def test_dpr(dpr_val):
    print(f"\n--- Testing at Device Pixel Ratio: {dpr_val} ---")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # Using iPad dimensions roughly (810x1080) for testing
        context = await browser.new_context(viewport={'width': 810, 'height': 1080}, device_scale_factor=dpr_val)
        page = await context.new_page()

        await page.goto("file:///c:/The_Knowledge_Galaxy/frontend/index.html")
        await page.wait_for_selector("canvas")
        # Ensure it renders properly
        await page.wait_for_timeout(2000)

        # Get all galaxy positions and names
        galaxies = await page.evaluate('''() => {
            const { CLUSTERS, canvas } = window.__APP_STATE;
            const w = canvas.getBoundingClientRect().width;
            const h = canvas.getBoundingClientRect().height;
            return CLUSTERS.map(c => ({
                label: c.label,
                x: c.rel[0] * w,
                y: c.rel[1] * h
            }));
        }''')
        
        print(f"Found {len(galaxies)} galaxies to test.")
        box = await page.locator("canvas").bounding_box()
        canvas_x = box['x']
        canvas_y = box['y']

        for g in galaxies:
            print(f"Testing drag for galaxy: '{g['label']}'")
            start_x = canvas_x + g['x']
            start_y = canvas_y + g['y']
            
            # Move mouse to the center of the galaxy
            await page.mouse.move(start_x, start_y)
            await page.wait_for_timeout(200)
            
            # Press down (pointerdown)
            await page.mouse.down()
            await page.wait_for_timeout(200)
            
            # Verify that `draggingCluster` is set to this galaxy
            dragging_label = await page.evaluate('''() => {
                // In index.html, `draggingCluster` is not exposed in __APP_STATE directly,
                // but the cursor will be "grabbing" if a cluster is successfully targeted.
                return window.__APP_STATE.canvas.style.cursor === 'grabbing';
            }''')
            
            if dragging_label:
                print("  -> Pointerdown hit successfully (cursor is 'grabbing').")
            else:
                print("  -> FAILED: Pointerdown missed the galaxy!")
            
            # Drag it by 50 pixels right and down
            await page.mouse.move(start_x + 50, start_y + 50, steps=10)
            await page.wait_for_timeout(200)
            
            # Check if it actually moved in state
            moved_pos = await page.evaluate(f'''() => {{
                const c = window.__APP_STATE.CLUSTERS.find(cl => cl.label === '{g["label"]}');
                return {{ rel_x: c.rel[0], rel_y: c.rel[1], manuallyPositioned: c.manuallyPositioned }};
            }}''')
            
            # Release mouse (pointerup)
            await page.mouse.up()
            await page.wait_for_timeout(200)

            # verify it moved relative to initial
            print(f"  -> Galaxy moved: {moved_pos['manuallyPositioned']} (New relative pos: {moved_pos['rel_x']:.3f}, {moved_pos['rel_y']:.3f})")

        await browser.close()

async def main():
    await test_dpr(2.0)
    await test_dpr(3.0)

if __name__ == "__main__":
    asyncio.run(main())
