/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Sun Nov 24 14:13:27 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0xc5325b28724b0a39cb7150f50f8c846a"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Sun Nov 24 14:13:27 2024"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 40, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 800, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  forward_lstm_output0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 800, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  bidirectional_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1600, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  global_max_pooling1d_pool_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 40, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1600, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_peephole_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 60, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  forward_lstm_kernel_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  forward_lstm_recurrent_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1600, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  forward_lstm_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 80, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2000, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  dense_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 50, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  dense_1_dense_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  backward_lstm_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  forward_lstm_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 140, AI_STATIC)

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_1_dense_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_scratch0, AI_STATIC,
  1, 0x0,
  AI_SHAPE_INIT(4, 1, 140, 1, 1), AI_STRIDE_INIT(4, 4, 4, 560, 560),
  1, &backward_lstm_scratch0_array, NULL)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  forward_lstm_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 140, 1, 1), AI_STRIDE_INIT(4, 4, 4, 560, 560),
  1, &forward_lstm_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  3, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 40), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &input_0_output_array, NULL)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_output0, AI_STATIC,
  4, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 40), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &backward_lstm_output0_array, NULL)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  forward_lstm_output0, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 20, 1, 40), AI_STRIDE_INIT(4, 4, 4, 80, 80),
  1, &forward_lstm_output0_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  bidirectional_output, AI_STATIC,
  6, 0x0,
  AI_SHAPE_INIT(4, 1, 40, 1, 40), AI_STRIDE_INIT(4, 4, 4, 160, 160),
  1, &bidirectional_output_array, NULL)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  global_max_pooling1d_pool_output, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 40, 1, 1), AI_STRIDE_INIT(4, 4, 4, 160, 160),
  1, &global_max_pooling1d_pool_output_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_output, AI_STATIC,
  8, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &dense_dense_output_array, NULL)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  dense_output, AI_STATIC,
  9, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &dense_output_array, NULL)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_output, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_1_dense_output_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_output, AI_STATIC,
  11, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4, 4),
  1, &dense_1_output_array, NULL)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_kernel, AI_STATIC,
  12, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 1), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &backward_lstm_kernel_array, NULL)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_recurrent, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 20, 80, 1, 1), AI_STRIDE_INIT(4, 4, 80, 6400, 6400),
  1, &backward_lstm_recurrent_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_peephole, AI_STATIC,
  14, 0x0,
  AI_SHAPE_INIT(4, 1, 1, 1, 60), AI_STRIDE_INIT(4, 4, 4, 240, 240),
  1, &backward_lstm_peephole_array, NULL)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  backward_lstm_bias, AI_STATIC,
  15, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 1), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &backward_lstm_bias_array, NULL)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  forward_lstm_kernel, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 1), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &forward_lstm_kernel_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  forward_lstm_recurrent, AI_STATIC,
  17, 0x0,
  AI_SHAPE_INIT(4, 20, 80, 1, 1), AI_STRIDE_INIT(4, 4, 80, 6400, 6400),
  1, &forward_lstm_recurrent_array, NULL)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  forward_lstm_bias, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 80, 1, 1), AI_STRIDE_INIT(4, 4, 4, 320, 320),
  1, &forward_lstm_bias_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_weights, AI_STATIC,
  19, 0x0,
  AI_SHAPE_INIT(4, 40, 50, 1, 1), AI_STRIDE_INIT(4, 4, 160, 8000, 8000),
  1, &dense_dense_weights_array, NULL)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  dense_dense_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 50, 1, 1), AI_STRIDE_INIT(4, 4, 4, 200, 200),
  1, &dense_dense_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  dense_1_dense_weights, AI_STATIC,
  21, 0x0,
  AI_SHAPE_INIT(4, 50, 1, 1, 1), AI_STRIDE_INIT(4, 4, 200, 200, 200),
  1, &dense_1_dense_weights_array, NULL)



/**  Layer declarations section  **********************************************/


AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_layer, 4,
  NL_TYPE, 0x0, NULL,
  nl, forward_sigmoid,
  &dense_1_chain,
  NULL, &dense_1_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_1_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_1_dense_weights, &dense_1_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_dense_layer, 4,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_1_dense_chain,
  NULL, &dense_1_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_layer, 2,
  NL_TYPE, 0x0, NULL,
  nl, forward_relu,
  &dense_chain,
  NULL, &dense_1_dense_layer, AI_STATIC, 
  .nl_params = NULL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_dense_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_max_pooling1d_pool_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &dense_dense_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &dense_dense_weights, &dense_dense_bias),
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_dense_layer, 2,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense,
  &dense_dense_chain,
  NULL, &dense_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  global_max_pooling1d_pool_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &bidirectional_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &global_max_pooling1d_pool_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  global_max_pooling1d_pool_layer, 1,
  POOL_TYPE, 0x0, NULL,
  pool, forward_mp,
  &global_max_pooling1d_pool_chain,
  NULL, &dense_dense_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(1, 40), 
  .pool_stride = AI_SHAPE_2D_INIT(1, 40), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  bidirectional_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &forward_lstm_output0, &backward_lstm_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &bidirectional_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  bidirectional_layer, 0,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &bidirectional_chain,
  NULL, &global_max_pooling1d_pool_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  forward_lstm_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &forward_lstm_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 9, &forward_lstm_kernel, &forward_lstm_recurrent, &backward_lstm_peephole, &forward_lstm_bias, NULL, NULL, NULL, NULL, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &forward_lstm_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  forward_lstm_layer, 0,
  LSTM_TYPE, 0x0, NULL,
  lstm, forward_lstm,
  &forward_lstm_chain,
  NULL, &bidirectional_layer, AI_STATIC, 
  .n_units = 20, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = false, 
  .reverse_seq = false, 
  .return_state = false, 
  .out_nl = nl_func_tanh_array_f32, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .cell_clip = 3e+38, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  backward_lstm_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &input_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &backward_lstm_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 9, &backward_lstm_kernel, &backward_lstm_recurrent, &backward_lstm_peephole, &backward_lstm_bias, NULL, NULL, NULL, NULL, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &backward_lstm_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  backward_lstm_layer, 0,
  LSTM_TYPE, 0x0, NULL,
  lstm, forward_lstm,
  &backward_lstm_chain,
  NULL, &forward_lstm_layer, AI_STATIC, 
  .n_units = 20, 
  .activation_nl = nl_func_tanh_array_f32, 
  .go_backwards = true, 
  .reverse_seq = true, 
  .return_state = false, 
  .out_nl = nl_func_tanh_array_f32, 
  .recurrent_nl = nl_func_sigmoid_array_f32, 
  .cell_clip = 3e+38, 
  .state = AI_HANDLE_PTR(NULL), 
  .init = AI_LAYER_FUNC(NULL), 
  .destroy = AI_LAYER_FUNC(NULL), 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 22724, 1, 1),
    22724, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 13360, 1, 1),
    13360, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_1_output),
  &backward_lstm_layer, 0x18a1b238, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 22724, 1, 1),
      22724, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 13360, 1, 1),
      13360, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &dense_1_output),
  &backward_lstm_layer, 0x18a1b238, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    backward_lstm_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    backward_lstm_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    forward_lstm_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    forward_lstm_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    input_0_output_array.data = AI_PTR(g_network_activations_map[0] + 6800);
    input_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6800);
    backward_lstm_output0_array.data = AI_PTR(g_network_activations_map[0] + 6960);
    backward_lstm_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 6960);
    forward_lstm_output0_array.data = AI_PTR(g_network_activations_map[0] + 10160);
    forward_lstm_output0_array.data_start = AI_PTR(g_network_activations_map[0] + 10160);
    bidirectional_output_array.data = AI_PTR(g_network_activations_map[0] + 560);
    bidirectional_output_array.data_start = AI_PTR(g_network_activations_map[0] + 560);
    global_max_pooling1d_pool_output_array.data = AI_PTR(g_network_activations_map[0] + 6960);
    global_max_pooling1d_pool_output_array.data_start = AI_PTR(g_network_activations_map[0] + 6960);
    dense_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 560);
    dense_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 560);
    dense_output_array.data = AI_PTR(g_network_activations_map[0] + 760);
    dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 760);
    dense_1_dense_output_array.data = AI_PTR(g_network_activations_map[0] + 560);
    dense_1_dense_output_array.data_start = AI_PTR(g_network_activations_map[0] + 560);
    dense_1_output_array.data = AI_PTR(g_network_activations_map[0] + 564);
    dense_1_output_array.data_start = AI_PTR(g_network_activations_map[0] + 564);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    backward_lstm_kernel_array.format |= AI_FMT_FLAG_CONST;
    backward_lstm_kernel_array.data = AI_PTR(g_network_weights_map[0] + 0);
    backward_lstm_kernel_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    backward_lstm_recurrent_array.format |= AI_FMT_FLAG_CONST;
    backward_lstm_recurrent_array.data = AI_PTR(g_network_weights_map[0] + 320);
    backward_lstm_recurrent_array.data_start = AI_PTR(g_network_weights_map[0] + 320);
    backward_lstm_peephole_array.format |= AI_FMT_FLAG_CONST;
    backward_lstm_peephole_array.data = AI_PTR(g_network_weights_map[0] + 6720);
    backward_lstm_peephole_array.data_start = AI_PTR(g_network_weights_map[0] + 6720);
    backward_lstm_bias_array.format |= AI_FMT_FLAG_CONST;
    backward_lstm_bias_array.data = AI_PTR(g_network_weights_map[0] + 6960);
    backward_lstm_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 6960);
    forward_lstm_kernel_array.format |= AI_FMT_FLAG_CONST;
    forward_lstm_kernel_array.data = AI_PTR(g_network_weights_map[0] + 7280);
    forward_lstm_kernel_array.data_start = AI_PTR(g_network_weights_map[0] + 7280);
    forward_lstm_recurrent_array.format |= AI_FMT_FLAG_CONST;
    forward_lstm_recurrent_array.data = AI_PTR(g_network_weights_map[0] + 7600);
    forward_lstm_recurrent_array.data_start = AI_PTR(g_network_weights_map[0] + 7600);
    forward_lstm_bias_array.format |= AI_FMT_FLAG_CONST;
    forward_lstm_bias_array.data = AI_PTR(g_network_weights_map[0] + 14000);
    forward_lstm_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 14000);
    dense_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 14320);
    dense_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 14320);
    dense_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 22320);
    dense_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22320);
    dense_1_dense_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_weights_array.data = AI_PTR(g_network_weights_map[0] + 22520);
    dense_1_dense_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 22520);
    dense_1_dense_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_dense_bias_array.data = AI_PTR(g_network_weights_map[0] + 22720);
    dense_1_dense_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 22720);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 146161,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x18a1b238,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 146161,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x18a1b238,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

