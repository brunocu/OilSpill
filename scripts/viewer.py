import os
import re
import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import rasterio
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import argparse
import numpy as np
import matplotlib.gridspec as gridspec

logging.getLogger("rasterio").setLevel(logging.ERROR)

class SARViewer:
    def __init__(self, master, show_predictions=False):
        self.master = master
        self.show_predictions = show_predictions
        self.master.title("Sentinel-1 Dual Polarization Viewer")
        # When the user closes the window, run self.on_closing
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Initially set folders to None (user picks them)
        self.oil_folder = None
        self.mask_folder = None
        self.pred_folder = None  # used only if predictions enabled
        self.image_files = []
        self.current_index = 0

        # Create figure based on mode
        if self.show_predictions:
            # Two-row structure: top row 2 subplots; bottom row 3 subplots.
            self.fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])
            self.axes_top = [self.fig.add_subplot(gs[0, i]) for i in range(2)]
            self.axes_bottom = [self.fig.add_subplot(gs[1, i]) for i in range(3)]
        else:
            # Two-row structure: top row 2 subplots; bottom row 1 subplot.
            self.fig = plt.figure(figsize=(12, 8))
            gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
            self.axes_top = [self.fig.add_subplot(gs[0, 0]), self.fig.add_subplot(gs[0, 1])]
            # Instead of duplicating the top, define a separate axis for the mask.
            self.ax_mask = self.fig.add_subplot(gs[1, 0])
        self.fig.tight_layout()

        # Embed the figure in Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Frame for the controls
        control_frame = tk.Frame(self.master)
        control_frame.pack(side=tk.BOTTOM, fill=tk.X)

        # Dropdown menu
        self.selected_image_var = tk.StringVar()
        self.dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.selected_image_var,
            values=self.image_files,
            state='readonly'
        )
        self.dropdown.bind('<<ComboboxSelected>>', self.on_dropdown_select)
        self.dropdown.pack(side=tk.LEFT, padx=5)

        # Navigation buttons
        self.prev_button = tk.Button(control_frame, text="Prev", command=self.show_prev)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(control_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.switch_button = tk.Button(control_frame, text="Switch Folder", command=self.select_folder)
        self.switch_button.pack(side=tk.LEFT, padx=5)

        logging.info("Initializing SARViewer and prompting for initial folder.")
        self.select_folder()

    def select_folder(self):
        """Prompt user to select an images folder. Then, use the sibling folder 'mask'."""
        logging.info("Opening folder selection dialog for images folder.")
        folder = filedialog.askdirectory(title="Select the images folder")
        if folder:
            logging.info(f"User selected images folder: {folder}")
            mask_folder = os.path.join(os.path.dirname(folder), "mask")
            logging.info(f"Images folder = {folder}, expecting mask folder = {mask_folder}")

            if not os.path.isdir(mask_folder):
                logging.error(f"Mask folder not found: {mask_folder}")
                messagebox.showerror("Error", f"Mask folder not found: {mask_folder}")
                return

            self.oil_folder = folder
            self.mask_folder = mask_folder

            if self.show_predictions:
                # Prompt user to select predictions folder (only for prediction mode)
                logging.info("Opening folder selection dialog for prediction masks.")
                pred_folder = filedialog.askdirectory(title="Select the predictions folder")
                if pred_folder and os.path.isdir(pred_folder):
                    self.pred_folder = pred_folder
                    logging.info(f"Prediction folder set to: {self.pred_folder}")
                else:
                    logging.error("Prediction folder not selected or invalid.")
                    messagebox.showerror("Error", "Prediction folder not selected or invalid.")
                    return

            self.load_images()
            if self.image_files:
                self.show_image(0)
            else:
                logging.info("No images found after loading.")
        else:
            logging.info("No folder was selected.")

    def load_images(self):
        """Load .tif/.tiff filenames from the selected source folder."""
        logging.info(f"Loading images from folder: {self.oil_folder}")
        if not self.oil_folder:
            logging.error("No source folder set before loading images.")
            return

        self.image_files = [
            f for f in os.listdir(self.oil_folder)
            if f.lower().endswith('.tif') or f.lower().endswith('.tiff')
        ]
        self.image_files.sort()
        logging.info(f"Found {len(self.image_files)} TIFF files in {self.oil_folder}.")
        self.current_index = 0

        # Update dropdown list
        self.dropdown['values'] = self.image_files
        if self.image_files:
            self.dropdown.current(0)
        else:
            self.dropdown.set("")

    def on_dropdown_select(self, event):
        selected_filename = self.selected_image_var.get()
        logging.info(f"Dropdown selected: {selected_filename}")
        if selected_filename in self.image_files:
            new_index = self.image_files.index(selected_filename)
            self.show_image(new_index)
        else:
            logging.warning(f"Selected filename {selected_filename} not in image list.")

    def show_prev(self):
        if not self.image_files:
            logging.info("No image files available for show_prev.")
            return
        new_index = (self.current_index - 1) % len(self.image_files)
        logging.info(f"Showing previous image: index {new_index}")
        self.show_image(new_index)

    def show_next(self):
        if not self.image_files:
            logging.info("No image files available for show_next.")
            return
        new_index = (self.current_index + 1) % len(self.image_files)
        logging.info(f"Showing next image: index {new_index}")
        self.show_image(new_index)

    def show_image(self, index):
        if not self.image_files:
            logging.info("No image files to show.")
            return
        
        self.master.config(cursor="watch")
        self.master.update_idletasks()

        self.current_index = index
        filename = self.image_files[self.current_index]
        logging.info(f"Displaying image at index {index}: {filename}")

        # Update dropdown
        self.selected_image_var.set(filename)

        if self.show_predictions:
            # Clear axes for predictions layout
            for ax in self.axes_top + self.axes_bottom:
                ax.clear()
        else:
            # Clear axes for non-prediction layout
            for ax in self.axes_top:
                ax.clear()
            self.ax_mask.clear()

        oil_path = os.path.join(self.oil_folder, filename)
        logging.info(f"Image path: {oil_path}")

        # Read the two-band source image
        try:
            with rasterio.open(oil_path) as src:
                vv = src.read(1)
                vh = src.read(2)
            logging.info("Loaded SAR data (VV, VH) successfully.")
        except Exception as e:
            logging.exception(f"Could not read source image: {oil_path}")
            messagebox.showerror("Error", f"Could not read source image:\n{oil_path}\n{e}")
            return

        if self.show_predictions:
            # Build ground truth mask filename: insert '_segmentation' before extension.
            base, ext = os.path.splitext(filename)
            gt_filename = f"{base}_segmentation{ext}"
            gt_path = os.path.join(self.mask_folder, gt_filename)
            logging.info(f"Ground truth mask path: {gt_path}")
            try:
                with rasterio.open(gt_path) as gt_src:
                    gt_mask = gt_src.read(1)
                logging.info("Loaded ground truth mask successfully.")
            except Exception as e:
                logging.exception(f"Could not read ground truth mask: {gt_path}")
                messagebox.showerror("Error", f"Could not read ground truth mask:\n{gt_path}\n{e}")
                return

            # Build predicted mask filename: add '_pred' suffix and use .png extension.
            pred_filename = f"{base}_pred.png"
            pred_path = os.path.join(self.pred_folder, pred_filename)
            logging.info(f"Predicted mask path: {pred_path}")
            try:
                with rasterio.open(pred_path) as pred_src:
                    pred_mask = pred_src.read(1)
                logging.info("Loaded predicted mask successfully.")
            except Exception as e:
                logging.exception(f"Could not read predicted mask: {pred_path}")
                messagebox.showerror("Error", f"Could not read predicted mask:\n{pred_path}\n{e}")
                return

            # Top row: show original images
            self.axes_top[0].imshow(vv, cmap='gray')
            self.axes_top[0].set_title("VV Band")
            self.axes_top[0].axis('off')
            self.axes_top[1].imshow(vh, cmap='gray')
            self.axes_top[1].set_title("VH Band")
            self.axes_top[1].axis('off')

            # Bottom row: show ground truth, prediction and composite
            self.axes_bottom[0].imshow(gt_mask, cmap='gray')
            self.axes_bottom[0].set_title("Ground Truth Mask")
            self.axes_bottom[0].axis('off')
            self.axes_bottom[1].imshow(pred_mask, cmap='gray')
            self.axes_bottom[1].set_title("Predicted Mask")
            self.axes_bottom[1].axis('off')
            # Create composite mask: blue is ground truth, red is prediction
            composite = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3), dtype=np.uint8)
            composite[..., 0] = (pred_mask * 255).astype(np.uint8)  # red
            composite[..., 2] = (gt_mask * 255).astype(np.uint8)    # blue
            self.axes_bottom[2].imshow(composite)
            self.axes_bottom[2].set_title("Composite Mask")
            self.axes_bottom[2].axis('off')
        else:
            # Non-prediction mode: read same-named mask
            mask_path = os.path.join(self.mask_folder, filename)
            logging.info(f"Mask path: {mask_path}")
            try:
                with rasterio.open(mask_path) as msk_src:
                    mask_data = msk_src.read(1)
                logging.info("Loaded mask data successfully.")
            except Exception as e:
                logging.exception(f"Could not read mask image: {mask_path}")
                messagebox.showerror("Error", f"Could not read mask image:\n{mask_path}\n{e}")
                return

            self.axes_top[0].imshow(vv, cmap='gray')
            self.axes_top[0].set_title("VV Band")
            self.axes_top[0].axis('off')
            self.axes_top[1].imshow(vh, cmap='gray')
            self.axes_top[1].set_title("VH Band")
            self.axes_top[1].axis('off')
            self.ax_mask.imshow(mask_data, cmap='gray')
            self.ax_mask.set_title("Mask")
            self.ax_mask.axis('off')

        self.fig.tight_layout()
        self.canvas.draw()
        logging.info("Finished drawing the selected image.")

        # Revert cursor back to default
        self.master.config(cursor="")

    def on_closing(self):
        # Close the matplotlib figure
        logging.info("Closing application.")
        plt.close(self.fig)
        # Stop the Tkinter main loop
        self.master.quit()
        self.master.destroy()

def main():
    parser = argparse.ArgumentParser(
        description="SAR Viewer Application that displays Sentinel-1 SAR images. "
    )
    parser.add_argument("--pred", action="store_true", 
                        help="Enable prediction view mode and load predicted masks (requires predictions folder)")
    args = parser.parse_args()

    logging.info("Starting SARViewer application.")
    root = tk.Tk()
    app = SARViewer(root, show_predictions=args.pred)
    root.mainloop()
    logging.info("SARViewer application has exited.")

if __name__ == "__main__":
    main()
