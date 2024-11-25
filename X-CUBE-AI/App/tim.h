#ifndef __TIM_H
#define __TIM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32l4xx_hal.h" // Adjust according to your STM32 family (e.g., stm32f4xx_hal.h)

// Declaration of the TIM handle
extern TIM_HandleTypeDef htim16;

// Function prototype for TIM16 initialization
void MX_TIM16_Init(void);

#ifdef __cplusplus
}
#endif

#endif /* __TIM16_H */
