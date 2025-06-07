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

	internal void setContextKey(int key) {
		_contextKey = key;
		Native.set_shared_context_key(key);
	}

	internal int getContextkey() {
		return _contextKey;
	}

	internal Image<ColorRGB>[] convertToImages(int width, int height, int batch) {
		Native.sd_image_t* images = Native.create_images(width, height);
		return ImageHelper.ToImages(images, batch);
	}

	internal bool convert_to_tensors(byte[][] imageData, int width, int height, int batch) {
		if (imageData.Length == 0)
			return false;

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
		}
		finally {
			for (int i = 0; i < handles.Length; i++) {
				if (handles[i].IsAllocated)
					handles[i].Free();
			}
		}

		// c#
		return true;
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
