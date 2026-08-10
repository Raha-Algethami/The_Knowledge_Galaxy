import asyncio
from playwright.async_api import async_playwright
import math
import json

async def main():
    async with async_playwright() as p:
        # Launch browser
        browser = await p.chromium.launch(headless=True)
        # Create a context with device_scale_factor=2 to simulate high-DPI (Retina)
        context = await browser.new_context(
            viewport={'width': 1200, 'height': 800},
            device_scale_factor=2.0
        )
        page = await context.new_page()

        print("Navigating to local index.html ...")
        await page.goto("file:///c:/The_Knowledge_Galaxy/frontend/index.html")
        
        # Wait for the canvas to be ready
        await page.wait_for_selector("canvas")
        await page.wait_for_timeout(2000) # wait for intro animation to pass slightly
        
        # We need to find the coordinates of a shooter star.
        # We can execute JS in the page context to get the current position of a shooter.
        print("Extracting shooter coordinates...")
        shooter_pos = await page.evaluate('''() => {
            const currentStudents = SUBJECTS[activeSubjectKey].students;
            const shooter = currentStudents.find(s => s.isShooter === true);
            if (!shooter) return null;
            return {
                x: shooter.px != null ? shooter.px : shooter.x,
                y: shooter.py != null ? shooter.py : shooter.y,
                name: shooter.name
            };
        }''')

        if not shooter_pos:
            print("No shooter found!")
            await browser.close()
            return

        print(f"Shooter '{shooter_pos['name']}' found at ({shooter_pos['x']}, {shooter_pos['y']})")
        
        # The coordinates x, y from JS are relative to the canvas.
        # We need to offset them by the canvas bounding box.
        box = await page.locator("canvas").bounding_box()
        click_x = box['x'] + shooter_pos['x']
        click_y = box['y'] + shooter_pos['y']
        
        print(f"Clicking at screen coordinates ({click_x}, {click_y})")
        
        # Simulate a touch/click by dispatching pointer events to accurately test touch/hover
        # Playwright's page.mouse.click() sends mouse events. To emulate touch, we could use page.touchscreen.
        # But mouse hover is also a pointer event. Let's just mouse move to the spot.
        await page.mouse.move(click_x, click_y)
        
        # We need to wait a frame for the canvas to render the tooltip.
        await page.wait_for_timeout(500)
        
        # Check if the tooltip is currently being drawn.
        # Since it's canvas, we can't inspect the DOM. We can evaluate JS to see if 'hovered' is set to the shooter.
        is_hovered = await page.evaluate('''() => {
            return hovered && hovered.isShooter === true;
        }''')
        
        if is_hovered:
            print("SUCCESS: Shooter star was successfully hovered/tapped on a DPR=2 screen!")
        else:
            print("FAILURE: Hover state was not triggered. Hit detection is still misaligned on high-DPI screens.")
            
        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
