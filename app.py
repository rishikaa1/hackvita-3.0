from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import cv2 as cv
import numpy as np
import os
from model import RoadSegmentationModel

app = FastAPI(title="Road Segmentation API")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

os.makedirs("static/uploads", exist_ok=True)
os.makedirs("static/outputs", exist_ok=True)
os.makedirs("static/debug", exist_ok=True)

try:
    model = RoadSegmentationModel.load_model('final_unet_model.keras')
except Exception as e:
    raise RuntimeError(f"Failed to load U-Net model: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """Endpoint to upload an image and get an annotated image with roads overlaid."""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    input_path = f"static/uploads/{file.filename}"
    base_name = file.filename.split('.')[0]
    
    try:
        # Save uploaded file
        with open(input_path, "wb") as f:
            f.write(await file.read())
        
        # Read the original image
        original_img = cv.imread(input_path)
        if original_img is None:
            raise ValueError("Failed to load original image")
        
        # Get predicted mask
        mask = RoadSegmentationModel.predict(model, input_path)
        if mask is None:
            raise ValueError("Prediction returned None")
        
        # Save raw mask for inspection
        raw_mask_path = f"static/debug/raw_mask_{base_name}.png"
        cv.imwrite(raw_mask_path, mask)
        
        # Create debug information
        debug_paths = {}
        debug_paths['raw_mask'] = f"/static/debug/raw_mask_{base_name}.png"
        
        # Print debug info
        print(f"Mask shape: {mask.shape}")
        print(f"Mask type: {mask.dtype}")
        print(f"Mask min/max: {mask.min()}/{mask.max()}")
        
        # Save grayscale version (for clear visualization)
        if len(mask.shape) > 2:
            grayscale = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        else:
            grayscale = mask
        gray_path = f"static/debug/gray_{base_name}.png"
        cv.imwrite(gray_path, grayscale)
        debug_paths['grayscale'] = f"/static/debug/gray_{base_name}.png"
        
        # Try both binary thresholding methods
        _, binary1 = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY)
        binary1_path = f"static/debug/binary1_{base_name}.png"
        cv.imwrite(binary1_path, binary1)
        debug_paths['binary1'] = f"/static/debug/binary1_{base_name}.png"
        
        # Try inverse binary threshold (in case mask is inverted)
        _, binary2 = cv.threshold(grayscale, 127, 255, cv.THRESH_BINARY_INV)
        binary2_path = f"static/debug/binary2_{base_name}.png"
        cv.imwrite(binary2_path, binary2)
        debug_paths['binary2'] = f"/static/debug/binary2_{base_name}.png"
        
        # Try different colored overlays
        # 1. Yellow overlay with binary1
        result1 = original_img.copy()
        yellow_mask = np.zeros_like(original_img)
        yellow_mask[binary1 == 255] = (0, 255, 255)  # Yellow
        yellow_path = f"static/debug/yellow_{base_name}.png"
        cv.addWeighted(yellow_mask, 0.5, result1, 0.5, 0, result1)
        cv.imwrite(yellow_path, result1)
        debug_paths['yellow'] = f"/static/debug/yellow_{base_name}.png"
        
        # 2. Blue overlay with binary1
        result2 = original_img.copy()
        blue_mask = np.zeros_like(original_img)
        blue_mask[binary1 == 255] = (255, 0, 0)  # Blue
        blue_path = f"static/debug/blue_{base_name}.png"
        cv.addWeighted(blue_mask, 0.5, result2, 0.5, 0, result2)
        cv.imwrite(blue_path, result2)
        debug_paths['blue'] = f"/static/debug/blue_{base_name}.png"
        
        # 3. Red overlay with binary2 (inverted mask)
        result3 = original_img.copy()
        red_mask = np.zeros_like(original_img)
        red_mask[binary2 == 255] = (0, 0, 255)  # Red
        red_path = f"static/debug/red_{base_name}.png"
        cv.addWeighted(red_mask, 0.5, result3, 0.5, 0, result3)
        cv.imwrite(red_path, result3)
        debug_paths['red'] = f"/static/debug/red_{base_name}.png"
        
        # 4. Green overlay with raw mask directly
        result4 = original_img.copy()
        green_mask = np.zeros_like(original_img)
        if len(mask.shape) > 2:
            mask_channel = mask[:,:,0]  # Take first channel if multi-channel
        else:
            mask_channel = mask
        green_mask[mask_channel > 0] = (0, 255, 0)  # Green on any non-zero value
        green_path = f"static/debug/green_{base_name}.png"
        cv.addWeighted(green_mask, 0.5, result4, 0.5, 0, result4)
        cv.imwrite(green_path, result4)
        debug_paths['green'] = f"/static/debug/green_{base_name}.png"
        
        # Final output (using blue overlay as default)
        output_path = f"static/outputs/annotated_{base_name}.png"
        cv.imwrite(output_path, result2)  # Use blue overlay as final result
        
        # Return all paths for diagnosis
        response = {
            "original": f"/static/uploads/{file.filename}",
            "annotated": f"/static/outputs/annotated_{base_name}.png",
            "debug": debug_paths
        }
        return JSONResponse(content=response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)