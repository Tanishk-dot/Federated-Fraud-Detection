import fitz # PyMuPDF
import requests
import io

print("Downloading paper...")
# Download FedGraphNN paper
url = "https://arxiv.org/pdf/2104.07145.pdf"
response = requests.get(url)
pdf_stream = io.BytesIO(response.content)

print("Extracting images...")
doc = fitz.open(stream=pdf_stream, filetype="pdf")
image_count = 0

for page_index in range(len(doc)):
    page = doc.load_page(page_index)
    image_list = page.get_images(full=True)
    
    for img_index, img in enumerate(image_list):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        
        # Filter out tiny icons/logos
        if len(image_bytes) > 20000:  
            filename = f"/Volumes/Untitled/federated-fraud-detection/paper_figure_p{page_index+1}_{img_index}.{image_ext}"
            with open(filename, "wb") as f:
                f.write(image_bytes)
            print(f"Saved {filename}")
            image_count += 1

print(f"Extracted {image_count} large images.")
