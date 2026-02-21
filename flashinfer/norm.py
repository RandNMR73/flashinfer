"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import inspect
from typing import Optional

import torch

from .api_logging import flashinfer_api
from .jit.norm import gen_norm_module
from .utils import device_support_pdl, register_custom_op, register_fake_op


@functools.cache
def _get_norm_module_cached():
    return gen_norm_module().build_and_load()


@torch.compiler.disable(recursive=False)
def get_norm_module():
    # Avoid tracing FlashInfer JIT/module loading internals (path/file locks/stat)
    # under torch.compile. The underlying kernel behavior is unchanged.
    return _get_norm_module_cached()


@flashinfer_api
def rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size) or 3D shape (batch_size, num_heads, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::rmsnorm", mutates_args=("out",))
def _rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::rmsnorm")
def _rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::rmsnorm_quant", mutates_args=("out",))
def rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * weight[i]``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor.
    input: torch.Tensor
        Input tensor, 2D shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: float
        Scale factor for quantization.
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Normalized tensor, 2D shape (batch_size, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().rmsnorm_quant(out, input, weight, scale, eps, enable_pdl)


@register_fake_op("flashinfer::rmsnorm_quant")
def _rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::fused_add_rmsnorm", mutates_args=("input", "residual"))
def fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::fused_add_rmsnorm")
def _fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::fused_add_rmsnorm_quant", mutates_args=("out", "residual")
)
def fused_add_rmsnorm_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * weight[i]``

    Parameters
    ----------
    out: torch.Tensor
        The output tensor, will quantize the output to the dtype of this tensor.
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    scale: float
        Scale factor for quantization.
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().fused_add_rmsnorm_quant(
        out, input, residual, weight, scale, eps, enable_pdl
    )


@register_fake_op("flashinfer::fused_add_rmsnorm_quant")
def _fused_add_rmsnorm_quant_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: float,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
def gemma_rmsnorm(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""Gemma-style root mean square normalization.

    ``out[i] = (input[i] / RMS(input)) * (weight[i] + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    out: Optional[torch.Tensor]
        The output tensor, if specified, the kernel will update this tensor inplace.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_

    Returns
    -------
    output: torch.Tensor
        Gemma Normalized tensor, shape (batch_size, hidden_size).
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    if out is None:
        out = torch.empty_like(input)
    _gemma_rmsnorm(out, input, weight, eps, enable_pdl)
    return out


@register_custom_op("flashinfer::gemma_rmsnorm", mutates_args=("out",))
def _gemma_rmsnorm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_rmsnorm(out, input, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_rmsnorm")
def _gemma_rmsnorm_fake(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: Optional[bool],
) -> None:
    pass


@flashinfer_api
@register_custom_op(
    "flashinfer::gemma_fused_add_rmsnorm", mutates_args=("input", "residual")
)
def gemma_fused_add_rmsnorm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    r"""Gemma-style fused add root mean square normalization.

    Step 1:
    ``residual[i] += input[i]``

    Step 2:
    ``input[i] = (residual[i] / RMS(residual)) * (weight + 1)``

    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size).
    residual: torch.Tensor
        Residual tensor, shape (batch_size, hidden_size).
    weight: torch.Tensor
        Weight tensor, shape (hidden_size,).
    eps: float
        Epsilon for numerical stability.
    enable_pdl: bool
        Whether to enable `programmatic dependent launch
        <https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization>`_
    """
    if enable_pdl is None:
        enable_pdl = device_support_pdl(input.device)
    get_norm_module().gemma_fused_add_rmsnorm(input, residual, weight, eps, enable_pdl)


@register_fake_op("flashinfer::gemma_fused_add_rmsnorm")
def _gemma_fused_add_rmsnorm_fake(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    enable_pdl: Optional[bool] = None,
) -> None:
    pass


@flashinfer_api
@register_custom_op("flashinfer::layernorm", mutates_args=())
def layernorm(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Layer normalization.
    Parameters
    ----------
    input: torch.Tensor
        Input tensor, shape (batch_size, hidden_size). Need to be bfloat16.
    gemma: torch.Tensor
        Gemma tensor, shape (hidden_size,). Need to be float32.
    beta: torch.Tensor
        Beta tensor, shape (hidden_size,). Need to be float32.
    eps: float
        Epsilon for numerical stability.

    Returns
    -------
    output: torch.Tensor
        Layer Normalized tensor, shape (batch_size, hidden_size). Same dtype as input.
    """
    out = torch.empty_like(input)
    get_norm_module().layernorm(out, input, gemma, beta, eps)
    return out


@register_fake_op("flashinfer::layernorm")
def _layernorm_fake(
    input: torch.Tensor,
    gemma: torch.Tensor,
    beta: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    b, k = input.shape
    return input.new_empty([b, k])


# CuTe-DSL fused RMSNorm + FP4 Quantization kernels
# These require CuTe-DSL to be available and SM100+ (Blackwell) GPUs
try:
    from .cute_dsl import (
        add_rmsnorm_fp4quant as _raw_add_rmsnorm_fp4quant,
    )
    from .cute_dsl import (
        rmsnorm_fp4quant as _raw_rmsnorm_fp4quant,
    )
except ImportError:
    _raw_rmsnorm_fp4quant = None  # type: ignore[misc,assignment]
    _raw_add_rmsnorm_fp4quant = None  # type: ignore[misc,assignment]
    rmsnorm_fp4quant = None  # type: ignore[misc,assignment]
    add_rmsnorm_fp4quant = None  # type: ignore[misc,assignment]
else:
    def _scale_format_to_code(scale_format: str | None) -> int:
        if scale_format is None:
            return 0
        if scale_format == "e4m3":
            return 1
        if scale_format == "ue8m0":
            return 2
        raise ValueError(f"Unsupported scale_format: {scale_format}")


    def _code_to_scale_format(scale_format_code: int) -> str | None:
        if scale_format_code == 0:
            return None
        if scale_format_code == 1:
            return "e4m3"
        if scale_format_code == 2:
            return "ue8m0"
        raise ValueError(f"Unsupported scale_format_code: {scale_format_code}")


    @functools.cache
    def _raw_add_rmsnorm_fp4quant_supported_kwargs() -> frozenset[str]:
        try:
            return frozenset(
                inspect.signature(_raw_add_rmsnorm_fp4quant).parameters.keys()  # type: ignore[arg-type]
            )
        except Exception:
            return frozenset()


    def _call_raw_add_rmsnorm_fp4quant_compat(**kwargs):
        supported_kwargs = _raw_add_rmsnorm_fp4quant_supported_kwargs()
        if not supported_kwargs:
            return _raw_add_rmsnorm_fp4quant(**kwargs)  # type: ignore[misc]
        filtered_kwargs = {
            key: value for key, value in kwargs.items() if key in supported_kwargs
        }
        return _raw_add_rmsnorm_fp4quant(**filtered_kwargs)  # type: ignore[misc]


    @torch.library.custom_op(
        "flashinfer_fp4::rmsnorm_fp4quant",
        mutates_args=(),
        device_types="cuda",
    )
    def _rmsnorm_fp4quant_op(
        input: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        block_size: int,
        scale_format_code: int,
        is_sf_swizzled_layout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_scale = global_scale.to(
            device=input.device, dtype=torch.float32
        ).reshape(1)
        return _raw_rmsnorm_fp4quant(  # type: ignore[misc]
            input=input,
            weight=weight,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            scale_format=_code_to_scale_format(scale_format_code),
            is_sf_swizzled_layout=is_sf_swizzled_layout,
        )


    @_rmsnorm_fp4quant_op.register_fake
    def _rmsnorm_fp4quant_op_fake(
        input: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        block_size: int,
        scale_format_code: int,
        is_sf_swizzled_layout: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del weight, global_scale, eps
        hidden_size = input.shape[-1]
        y_fp4 = torch.empty(
            (*input.shape[:-1], hidden_size // 2),
            dtype=torch.float4_e2m1fn_x2,
            device=input.device,
        )
        scale_format = _code_to_scale_format(scale_format_code)
        actual_scale_format = (
            scale_format if scale_format else ("ue8m0" if block_size == 32 else "e4m3")
        )
        scale_dtype = (
            torch.uint8 if actual_scale_format == "ue8m0" else torch.float8_e4m3fn
        )
        if is_sf_swizzled_layout:
            rows = input.shape[0] if input.dim() == 2 else input.shape[0] * input.shape[1]
            num_sf_blocks_per_row = hidden_size // block_size
            num_m_tiles = (rows + 127) // 128
            num_k_tiles = (num_sf_blocks_per_row + 3) // 4
            block_scale = torch.empty(
                (num_m_tiles * num_k_tiles * 512,),
                dtype=scale_dtype,
                device=input.device,
            )
        else:
            block_scale = torch.empty(
                (*input.shape[:-1], hidden_size // block_size),
                dtype=scale_dtype,
                device=input.device,
            )
        return y_fp4, block_scale


    @torch.library.custom_op(
        "flashinfer_fp4::add_rmsnorm_fp4quant",
        mutates_args=(),
        device_types="cuda",
    )
    def _add_rmsnorm_fp4quant_op(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        block_size: int,
        scale_format_code: int,
        is_sf_swizzled_layout: bool,
        output_both_sf_layouts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        global_scale = global_scale.to(
            device=input.device, dtype=torch.float32
        ).reshape(1)
        result = _call_raw_add_rmsnorm_fp4quant_compat(
            input=input,
            residual=residual,
            weight=weight,
            global_scale=global_scale,
            eps=eps,
            block_size=block_size,
            scale_format=_code_to_scale_format(scale_format_code),
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            output_both_sf_layouts=output_both_sf_layouts,
        )
        if output_both_sf_layouts and len(result) == 3:
            y_fp4, block_scale, block_scale_unswizzled = result
            return y_fp4, block_scale, block_scale_unswizzled
        y_fp4, block_scale = result
        return y_fp4, block_scale, torch.empty(
            (0,), dtype=block_scale.dtype, device=block_scale.device
        )


    @_add_rmsnorm_fp4quant_op.register_fake
    def _add_rmsnorm_fp4quant_op_fake(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        global_scale: torch.Tensor,
        eps: float,
        block_size: int,
        scale_format_code: int,
        is_sf_swizzled_layout: bool,
        output_both_sf_layouts: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del residual, weight, global_scale, eps
        hidden_size = input.shape[-1]
        y_fp4 = torch.empty(
            (*input.shape[:-1], hidden_size // 2),
            dtype=torch.float4_e2m1fn_x2,
            device=input.device,
        )
        scale_format = _code_to_scale_format(scale_format_code)
        actual_scale_format = (
            scale_format if scale_format else ("ue8m0" if block_size == 32 else "e4m3")
        )
        scale_dtype = (
            torch.uint8 if actual_scale_format == "ue8m0" else torch.float8_e4m3fn
        )
        use_swizzled = is_sf_swizzled_layout or output_both_sf_layouts
        if use_swizzled:
            rows = input.shape[0] if input.dim() == 2 else input.shape[0] * input.shape[1]
            num_sf_blocks_per_row = hidden_size // block_size
            num_m_tiles = (rows + 127) // 128
            num_k_tiles = (num_sf_blocks_per_row + 3) // 4
            block_scale = torch.empty(
                (num_m_tiles * num_k_tiles * 512,),
                dtype=scale_dtype,
                device=input.device,
            )
        else:
            block_scale = torch.empty(
                (*input.shape[:-1], hidden_size // block_size),
                dtype=scale_dtype,
                device=input.device,
            )
        block_scale_unswizzled = torch.empty(
            (*input.shape[:-1], hidden_size // block_size),
            dtype=scale_dtype,
            device=input.device,
        )
        return y_fp4, block_scale, block_scale_unswizzled


    @flashinfer_api
    def rmsnorm_fp4quant(
        input: torch.Tensor,
        weight: torch.Tensor,
        y_fp4: torch.Tensor | None = None,
        block_scale: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
        eps: float = 1e-6,
        block_size: int = 16,
        scale_format: str | None = None,
        is_sf_swizzled_layout: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if y_fp4 is not None or block_scale is not None:
            return _raw_rmsnorm_fp4quant(  # type: ignore[misc]
                input=input,
                weight=weight,
                y_fp4=y_fp4,
                block_scale=block_scale,
                global_scale=global_scale,
                eps=eps,
                block_size=block_size,
                scale_format=scale_format,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
            )
        if global_scale is None:
            global_scale = torch.ones(1, dtype=torch.float32, device=input.device)
        return torch.ops.flashinfer_fp4.rmsnorm_fp4quant(
            input,
            weight,
            global_scale,
            eps,
            block_size,
            _scale_format_to_code(scale_format),
            is_sf_swizzled_layout,
        )


    @flashinfer_api
    def add_rmsnorm_fp4quant(
        input: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        y_fp4: torch.Tensor | None = None,
        block_scale: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
        eps: float = 1e-6,
        block_size: int = 16,
        scale_format: str | None = None,
        is_sf_swizzled_layout: bool = False,
        output_both_sf_layouts: bool = False,
        block_scale_unswizzled: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_fp4 is not None or block_scale is not None or block_scale_unswizzled is not None:
            result = _call_raw_add_rmsnorm_fp4quant_compat(
                input=input,
                residual=residual,
                weight=weight,
                y_fp4=y_fp4,
                block_scale=block_scale,
                global_scale=global_scale,
                eps=eps,
                block_size=block_size,
                scale_format=scale_format,
                is_sf_swizzled_layout=is_sf_swizzled_layout,
                output_both_sf_layouts=output_both_sf_layouts,
                block_scale_unswizzled=block_scale_unswizzled,
            )
            if output_both_sf_layouts and len(result) == 2:
                y_fp4_out, block_scale_out = result
                return (
                    y_fp4_out,
                    block_scale_out,
                    torch.empty(
                        (0,),
                        dtype=block_scale_out.dtype,
                        device=block_scale_out.device,
                    ),
                )
            return result
        if global_scale is None:
            global_scale = torch.ones(1, dtype=torch.float32, device=input.device)
        y_fp4, block_scale, block_scale_unswizzled = torch.ops.flashinfer_fp4.add_rmsnorm_fp4quant(
            input,
            residual,
            weight,
            global_scale,
            eps,
            block_size,
            _scale_format_to_code(scale_format),
            is_sf_swizzled_layout,
            output_both_sf_layouts,
        )
        if output_both_sf_layouts:
            return y_fp4, block_scale, block_scale_unswizzled
        return y_fp4, block_scale
