import re

with open('qwen_vl_test.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the start and end markers
start = content.find('    print_header("TEST 3: Image Encoding (Vision Domain)")')
end = content.find('    # -' * 1, content.find('TEST 4'))
end = content.find('    # ---', start + 100)

if start == -1:
    print("Could not find TEST 3 start")
    exit(1)

new_section = '''    print_header("TEST 3: Image Encoding (Vision Domain)")
    print("  Generating local test images with PIL...")
    try:
        from PIL import Image
        import os
        img_warm = Image.new("RGB", (224, 224), color=(220, 80, 60))
        img_cool = Image.new("RGB", (224, 224), color=(60, 80, 220))
        img_warm.save("test_warm.jpg")
        img_cool.save("test_cool.jpg")
        print("  Created test_warm.jpg (red) and test_cool.jpg (blue)")
        t0 = time.perf_counter()
        inputs = [
            "test_warm.jpg",
            "test_cool.jpg",
            "A warm red sunset glowing over the horizon.",
            "A cool blue ocean stretching to the horizon.",
        ]
        img_labels = ["Warm image", "Cool image", "Warm text", "Cool text"]
        with torch.no_grad():
            img_embeddings = model.encode(
                inputs,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            if not isinstance(img_embeddings, torch.Tensor):
                img_embeddings = torch.tensor(img_embeddings)
            img_embeddings = img_embeddings.to(device).float()
        elapsed = time.perf_counter() - t0
        print(f"  Encoded 2 images + 2 text in {elapsed:.2f}s")
        print(f"  Shape: {tuple(img_embeddings.shape)}")
        warm_img  = img_embeddings[0]
        cool_img  = img_embeddings[1]
        warm_text = img_embeddings[2]
        cool_text = img_embeddings[3]
        emb_list = [warm_img, cool_img, warm_text, cool_text]
        print("  Cross-modal similarity matrix:")
        for lbl, ei in zip(img_labels, emb_list):
            row = f"    {lbl:<14}"
            for ej in emb_list:
                row += f"  {cosine_sim(ei, ej):.4f}"
            print(row)
        warm_cross = cosine_sim(warm_img, warm_text)
        cool_cross = cosine_sim(warm_img, cool_text)
        print(f"  Warm image to Warm text : {warm_cross:.4f}")
        print(f"  Warm image to Cool text : {cool_cross:.4f}")
        result = "Working" if warm_cross > cool_cross else "Not aligned"
        print(f"  Cross-modal alignment   : {result}")
        os.remove("test_warm.jpg")
        os.remove("test_cool.jpg")
    except Exception as e:
        print(f"  Image test failed: {e}")
        import traceback
        traceback.print_exc()

'''

content = content[:start] + new_section + content[end:]

with open('qwen_vl_test.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("SUCCESS")
