using System;
using HPPH;
using System.Runtime.InteropServices;
using JetBrains.Annotations;

namespace StableDiffusion.NET;

public sealed unsafe class DiffusionModel : IDisposable
{
	#region Properties & Fields

	private bool _disposed;

	public DiffusionModelParameter ModelParameter { get; }

	private Native.API_ModelComponents* _nativeComponents;
	public ModelComponents components;

	private Native.sd_ctx_t* _ctx;
	private Shared _shared;
	#endregion

	#region Constructors

	public void setShared(Shared shared)
	{
		ArgumentNullException.ThrowIfNull(shared, nameof(shared));
		_shared = shared;
	}

	public DiffusionModel(DiffusionModelParameter modelParameter)
	{
		ArgumentNullException.ThrowIfNull(modelParameter, nameof(modelParameter));
		modelParameter.Validate();
		this.ModelParameter = modelParameter;

		// Initialize();
	}

	~DiffusionModel() => Dispose();

	#endregion

	#region Methods

	public void Initialize() {

		_ctx = Native.new_sd_ctx(_shared._getSharedData(),
								 ModelParameter.ModelPath,
								 ModelParameter.ClipLPath,
								 ModelParameter.ClipGPath,
								 ModelParameter.T5xxlPath,
								 ModelParameter.DiffusionModelPath,
								 ModelParameter.VaePath,
								 ModelParameter.TaesdPath,
								 ModelParameter.ControlNetPath,
								 ModelParameter.LoraModelDirectory,
								 ModelParameter.EmbeddingsDirectory,
								 ModelParameter.StackedIdEmbeddingsDirectory,
								 ModelParameter.VaeDecodeOnly,
								 ModelParameter.VaeTiling,
								 false,
								 ModelParameter.ThreadCount,
								 ModelParameter.Quantization,
								 ModelParameter.RngType,
								 ModelParameter.Schedule,
								 ModelParameter.KeepClipOnCPU,
								 ModelParameter.KeepControlNetOnCPU,
								 ModelParameter.KeepVaeOnCPU,
								 ModelParameter.FlashAttention);

		if (_shared == null) throw new NullReferenceException("Failed to initialize shared data.");
		if (_ctx == null) throw new NullReferenceException("Failed to initialize diffusion-model.");

		components = createComponents(_ctx);
	}

	internal ModelComponents createComponents(Native.sd_ctx_t* ctx) {
		_nativeComponents = Native.get_model_components(ctx);

		var vae = _nativeComponents->first_stage_model;
		var clip = _nativeComponents->cond_stage_model;

		// Validate the native components
		if (vae == null)
			throw new NullReferenceException("VAE component is null.");
		else if (clip == null)
			throw new NullReferenceException("CLIP component is null.");
		
		return new ModelComponents(
			new AutoEncoderKL(vae),
			new Conditioner(clip)
		);
	}

	public DiffusionParameter GetDefaultParameter() => ModelParameter.DiffusionModelType switch
	{
		DiffusionModelType.None => new DiffusionParameter(),
		DiffusionModelType.StableDiffusion => DiffusionParameter.SDXLDefault,
		// DiffusionModelType.Flux => DiffusionParameter.FluxDefault,
		_ => throw new ArgumentOutOfRangeException()
	};


	public bool vaeProcess(TensorArray latents, bool decode) {
		ObjectDisposedException.ThrowIf(_disposed, this);

		return Native.vae_process(_shared._getSharedData(), _ctx, latents._getNativeArray(), latents.Count, decode);
	}

	public IImage<ColorRGB> TextToImage(string prompt, Tensor? init_latent, DiffusionParameter? parameter = null, bool latents_only=false)
	{
		parameter ??= GetDefaultParameter();

		ObjectDisposedException.ThrowIf(_disposed, this);
		ArgumentNullException.ThrowIfNull(prompt);

		parameter.Validate();

		Native.sd_image_t* result = null;
		if (parameter.ControlNet.IsEnabled)
		{
			if (parameter.ControlNet.Image is not IImage<ColorRGB> controlNetImage)
				controlNetImage = parameter.ControlNet.Image!.ConvertTo<ColorRGB>();

			fixed (byte* imagePtr = controlNetImage.ToRawArray())
			{
				if (parameter.ControlNet.CannyPreprocess)
				{
					Native.sd_image_t nativeControlNetImage = new()
					{
						width = (uint)controlNetImage.Width,
						height = (uint)controlNetImage.Height,
						channel = (uint)controlNetImage.ColorFormat.BytesPerPixel,
						data = Native.preprocess_canny(imagePtr,
													   parameter.Width,
													   parameter.Height,
													   parameter.ControlNet.CannyHighThreshold,
													   parameter.ControlNet.CannyLowThreshold,
													   parameter.ControlNet.CannyWeak,
													   parameter.ControlNet.CannyStrong,
													   parameter.ControlNet.CannyInverse)
					};

					result = Native.txt2img(_ctx,
											_shared._getSharedData(),
											prompt,
											parameter.NegativePrompt,
											parameter.ClipSkip,
											parameter.CfgScale,
											parameter.Guidance,
											parameter.Eta,
											parameter.Width,
											parameter.Height,
											parameter.SampleMethod,
											parameter.SampleSteps,
											parameter.Seed,
											parameter.BatchCount,
											&nativeControlNetImage,
											parameter.ControlNet.Strength,
											parameter.PhotoMaker.StyleRatio,
											parameter.PhotoMaker.NormalizeInput,
											parameter.PhotoMaker.InputIdImageDirectory,
											parameter.SkipLayers,
											parameter.SkipLayers.Length,
											parameter.SlgScale,
											parameter.SkipLayerStart,
											parameter.SkipLayerEnd,
											latents_only,
											init_latent._getNativeTensor());

					Marshal.FreeHGlobal((nint)nativeControlNetImage.data);
				}
				else
				{
					Native.sd_image_t nativeControlNetImage = new()
					{
						width = (uint)controlNetImage.Width,
						height = (uint)controlNetImage.Height,
						channel = (uint)controlNetImage.ColorFormat.BytesPerPixel,
						data = imagePtr
					};

					result = Native.txt2img(_ctx,
											_shared._getSharedData(),
											prompt,
											parameter.NegativePrompt,
											parameter.ClipSkip,
											parameter.CfgScale,
											parameter.Guidance,
											parameter.Eta,
											parameter.Width,
											parameter.Height,
											parameter.SampleMethod,
											parameter.SampleSteps,
											parameter.Seed,
											parameter.BatchCount,
											&nativeControlNetImage,
											parameter.ControlNet.Strength,
											parameter.PhotoMaker.StyleRatio,
											parameter.PhotoMaker.NormalizeInput,
											parameter.PhotoMaker.InputIdImageDirectory,
											parameter.SkipLayers,
											parameter.SkipLayers.Length,
											parameter.SlgScale,
											parameter.SkipLayerStart,
											parameter.SkipLayerEnd,
											latents_only,
											init_latent._getNativeTensor());
				}
			}
		}
		else
		{
			result = Native.txt2img(_ctx,
									_shared._getSharedData(),
									prompt,
									parameter.NegativePrompt,
									parameter.ClipSkip,
									parameter.CfgScale,
									parameter.Guidance,
									parameter.Eta,
									parameter.Width,
									parameter.Height,
									parameter.SampleMethod,
									parameter.SampleSteps,
									parameter.Seed,
									parameter.BatchCount,
									null,
									0,
									parameter.PhotoMaker.StyleRatio,
									parameter.PhotoMaker.NormalizeInput,
									parameter.PhotoMaker.InputIdImageDirectory,
									parameter.SkipLayers,
									parameter.SkipLayers.Length,
									parameter.SlgScale,
									parameter.SkipLayerStart,
									parameter.SkipLayerEnd,
									latents_only,
									init_latent._getNativeTensor());
		}
		
		if (latents_only)
		  return null;
		return ImageHelper.ToImage(result);
	}

	public IImage<ColorRGB> ImageToImage(string prompt, IImage image, DiffusionParameter? parameter = null)
	{
		parameter ??= GetDefaultParameter();

		ObjectDisposedException.ThrowIf(_disposed, this);
		ArgumentNullException.ThrowIfNull(prompt);
		ArgumentNullException.ThrowIfNull(image);

		parameter.Validate();

		if (image is not IImage<ColorRGB> refImage)
			refImage = image.ConvertTo<ColorRGB>();

		// DarthAffe 10.08.2024: Mask needs to be a 1 channel all max value image when it's not used - I really don't like this concept as it adds unnecessary allocations, but that's how it is :(
		Span<byte> maskBuffer = new byte[image.Width * image.Height];
		maskBuffer.Fill(byte.MaxValue);

		fixed (byte* maskPtr = maskBuffer)
		{
			Native.sd_image_t maskImage = new()
			{
				width = (uint)image.Width,
				height = (uint)image.Height,
				channel = 1,
				data = maskPtr
			};

			fixed (byte* imagePtr = refImage.AsRefImage())
				return ImageToImage(prompt, refImage.ToSdImage(imagePtr), maskImage, parameter);
		}
	}

	public IImage<ColorRGB> Inpaint(string prompt, IImage image, IImage mask, DiffusionParameter? parameter = null)
	{
		parameter ??= GetDefaultParameter();

		ObjectDisposedException.ThrowIf(_disposed, this);
		ArgumentNullException.ThrowIfNull(prompt);
		ArgumentNullException.ThrowIfNull(image);
		ArgumentNullException.ThrowIfNull(mask);

		if (image.Width != mask.Width) throw new ArgumentException("The mask needs to have the same with as the image.", nameof(mask));
		if (image.Height != mask.Height) throw new ArgumentException("The mask needs to have the same height as the image.", nameof(mask));

		parameter.Validate();

		if (image is not IImage<ColorRGB> refImage)
			refImage = image.ConvertTo<ColorRGB>();

		// DarthAffe 10.08.2024: HPPH does currently not support monochrome images, that's why we need to convert it here. We're going for the simple conversion as the source image is supposed to be monochrome anyway.
		Span<byte> maskBuffer = new byte[image.Width * image.Height];
		for (int y = 0; y < image.Height; y++)
			for (int x = 0; x < image.Width; x++)
			{
				IColor color = mask[x, y];
				maskBuffer[(image.Width * y) + x] = (byte)Math.Round((color.R + color.G + color.B) / 3.0);
			}

		fixed (byte* maskPtr = maskBuffer)
		{
			Native.sd_image_t maskImage = new()
			{
				width = (uint)image.Width,
				height = (uint)image.Height,
				channel = 1,
				data = maskPtr
			};

			fixed (byte* imagePtr = refImage.AsRefImage())
				return ImageToImage(prompt, refImage.ToSdImage(imagePtr), maskImage, parameter);
		}

	}

	private IImage<ColorRGB> ImageToImage(string prompt, Native.sd_image_t image, Native.sd_image_t mask, DiffusionParameter parameter)
	{
		ObjectDisposedException.ThrowIf(_disposed, this);
		ArgumentNullException.ThrowIfNull(prompt);

		parameter.Validate();

		Native.sd_image_t* result;
		if (parameter.ControlNet.IsEnabled)
		{
			if (parameter.ControlNet.Image is not IImage<ColorRGB> controlNetImage)
				controlNetImage = parameter.ControlNet.Image!.ConvertTo<ColorRGB>();

			fixed (byte* imagePtr = controlNetImage.ToRawArray())
			{
				if (parameter.ControlNet.CannyPreprocess)
				{
					Native.sd_image_t nativeControlNetImage = new()
					{
						width = (uint)controlNetImage.Width,
						height = (uint)controlNetImage.Height,
						channel = (uint)controlNetImage.ColorFormat.BytesPerPixel,
						data = Native.preprocess_canny(imagePtr,
													   parameter.Width,
													   parameter.Height,
													   parameter.ControlNet.CannyHighThreshold,
													   parameter.ControlNet.CannyLowThreshold,
													   parameter.ControlNet.CannyWeak,
													   parameter.ControlNet.CannyStrong,
													   parameter.ControlNet.CannyInverse)
					};

					result = Native.img2img(_ctx,
											image,
											mask,
											prompt,
											parameter.NegativePrompt,
											parameter.ClipSkip,
											parameter.CfgScale,
											parameter.Guidance,
											parameter.Width,
											parameter.Height,
											parameter.SampleMethod,
											parameter.SampleSteps,
											parameter.Strength,
											parameter.Seed,
											1,
											&nativeControlNetImage,
											parameter.ControlNet.Strength,
											parameter.PhotoMaker.StyleRatio,
											parameter.PhotoMaker.NormalizeInput,
											parameter.PhotoMaker.InputIdImageDirectory,
											parameter.SkipLayers,
											parameter.SkipLayers.Length,
											parameter.SlgScale,
											parameter.SkipLayerStart,
											parameter.SkipLayerEnd);

					Marshal.FreeHGlobal((nint)nativeControlNetImage.data);
				}
				else
				{
					Native.sd_image_t nativeControlNetImage = new()
					{
						width = (uint)parameter.ControlNet.Image.Width,
						height = (uint)parameter.ControlNet.Image.Height,
						channel = (uint)parameter.ControlNet.Image.ColorFormat.BytesPerPixel,
						data = imagePtr
					};

					result = Native.img2img(_ctx,
											image,
											mask,
											prompt,
											parameter.NegativePrompt,
											parameter.ClipSkip,
											parameter.CfgScale,
											parameter.Guidance,
											parameter.Width,
											parameter.Height,
											parameter.SampleMethod,
											parameter.SampleSteps,
											parameter.Strength,
											parameter.Seed,
											1,
											&nativeControlNetImage,
											parameter.ControlNet.Strength,
											parameter.PhotoMaker.StyleRatio,
											parameter.PhotoMaker.NormalizeInput,
											parameter.PhotoMaker.InputIdImageDirectory,
											parameter.SkipLayers,
											parameter.SkipLayers.Length,
											parameter.SlgScale,
											parameter.SkipLayerStart,
											parameter.SkipLayerEnd);
				}
			}
		}
		else
		{
			result = Native.img2img(_ctx,
									image,
									mask,
									prompt,
									parameter.NegativePrompt,
									parameter.ClipSkip,
									parameter.CfgScale,
									parameter.Guidance,
									parameter.Width,
									parameter.Height,
									parameter.SampleMethod,
									parameter.SampleSteps,
									parameter.Strength,
									parameter.Seed,
									1,
									null,
									0,
									parameter.PhotoMaker.StyleRatio,
									parameter.PhotoMaker.NormalizeInput,
									parameter.PhotoMaker.InputIdImageDirectory,
									parameter.SkipLayers,
									parameter.SkipLayers.Length,
									parameter.SlgScale,
									parameter.SkipLayerStart,
									parameter.SkipLayerEnd);
		}

		return ImageHelper.ToImage(result);
	}

	public void Dispose()
	{
		if (_disposed) return;

		if (_ctx != null)
			Native.free_sd_ctx(_ctx);

		if (components != null)
			Native.free_model_components(_nativeComponents);

		GC.SuppressFinalize(this);
		_disposed = true;
	}

	#endregion
}

public sealed unsafe class ModelComponents
{
	// VAE
	public AutoEncoderKL Vae { get; }
	// CLIP
	public Conditioner Clip { get; }

	public ModelComponents(AutoEncoderKL vae, Conditioner clip)
	{
		Vae = vae;
		Clip = clip;
	}

	
}

public unsafe class AutoEncoderKL
{
	private Native._AutoEncoderKL* _nativeStruct;

	internal AutoEncoderKL(Native._AutoEncoderKL* nativeStruct)
	{
		Initialize(nativeStruct);
	}

	private void Initialize(Native._AutoEncoderKL* nativeStruct)
	{
		// Initialize the managed struct
		_nativeStruct = nativeStruct;
	}

	// Add methods and properties to interact with the native AutoEncoderKL
}

public unsafe class Conditioner
{
	private Native._Conditioner* _nativeStruct;

	internal Conditioner(Native._Conditioner* nativeStruct)
	{
		Initialize(nativeStruct);
	}

	private void Initialize(Native._Conditioner* nativeStruct)
	{
		// Initialize the managed struct
		_nativeStruct = nativeStruct;
	}

	// Add methods and properties to interact with the native Conditioner
}
