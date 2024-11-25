#include "tim.h"

// Define the TIM handle
TIM_HandleTypeDef htim16;

/**
  * @brief TIM16 Initialization Function
  * @param None
  * @retval None
  */
void MX_TIM16_Init(void)
{
  /* USER CODE BEGIN TIM16_Init 0 */

  /* USER CODE END TIM16_Init 0 */

  /* USER CODE BEGIN TIM16_Init 1 */

  /* USER CODE END TIM16_Init 1 */

  // Configure TIM16 instance
  htim16.Instance = TIM16;
  htim16.Init.Prescaler = 80 - 1; // Set prescaler to 80-1
  htim16.Init.CounterMode = TIM_COUNTERMODE_UP; // Count up mode
  htim16.Init.Period = 65536 - 1; // Auto-reload value
  htim16.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1; // No clock division
  htim16.Init.RepetitionCounter = 0; // Default repetition counter
  htim16.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE; // No preload

  // Initialize the TIM Base
  if (HAL_TIM_Base_Init(&htim16) != HAL_OK)
  {
    // Initialization Error
    Error_Handler();
  }

  /* USER CODE BEGIN TIM16_Init 2 */

  /* USER CODE END TIM16_Init 2 */
}
