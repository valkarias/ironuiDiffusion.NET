using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

using HPPH;

namespace StableDiffusion.NET;


public sealed unsafe class Shared {
	private int _contextKey;

	internal static Shared createSharedData() {
		return new Shared();
	}
	
	internal Shared() {
	}

	internal void setContext(int key) {
		_contextKey = key;
		Native.set_shared_context(key);
	}

	internal int getContextkey() {
		return _contextKey;
	}

	internal Image<ColorRGB>[] convertToImages(int width, int height, int batch) {
		Native.sd_image_t* images = Native.create_images(width, height);
		return ImageHelper.ToImages(images, batch);
	}

	internal int convertMaskToTensor(byte[] mask, int width, int height) {
		if (mask == null)
			return Constants.EMPTY_INDEX;

		var handle = new GCHandle();
		var pointer = new IntPtr();

		try {
			handle = GCHandle.Alloc(mask, GCHandleType.Pinned);
			pointer = handle.AddrOfPinnedObject();
			return Native.convert_mask_to_tensor((byte*)pointer, width, height);
		}
		finally {
			handle.Free();
		}

		// c#
		return Constants.EMPTY_INDEX;
	}

	internal int convertToTensors(byte[][] imageData, int width, int height, int batch) {
		if (imageData.Length == 0)
			return Constants.EMPTY_INDEX;

		var handles = new GCHandle[imageData.Length];
		var pointers = new IntPtr[imageData.Length];

		try {
			for (int i = 0; i < imageData.Length; i++) {
				handles[i] = GCHandle.Alloc(imageData[i], GCHandleType.Pinned);
				pointers[i] = handles[i].AddrOfPinnedObject();
			}

			fixed (IntPtr* ptr = pointers) {
				return Native.convert_to_tensors((byte**)ptr, width, height, imageData.Length);
			}
		} catch (Exception e) {
			throw e;
		} finally {
			for (int i = 0; i < handles.Length; i++) {
				if (handles[i].IsAllocated)
					handles[i].Free();
			}
		}

		// c#
		return Constants.EMPTY_INDEX;
	}

	//tensor + index
	//cache empty latent?
	internal int createEmptyTensor(int width, int height, int channels, int batch) {
		if (width <= 0 || height <= 0 || batch <= 0) {
			throw new ArgumentException("Width, height and batch must be greater than zero.");
		}
		
		return Native.create_empty_latent(width, height, channels, batch);
	}

	internal void cleanTensors(int key, TensorType type) {
		if (type == TensorType._ALL_)
			_contextKey = Constants.EMPTY_INDEX;
		Native.clean_tensors(key, type);
	}

	public void cleanUp() {
		_contextKey = Constants.EMPTY_INDEX;
		Native.clean_shared();
	}
}
