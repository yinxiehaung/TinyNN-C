#ifndef NN_LAYER_H
#define NN_LAYER_H

#include "matrix.h"
#include "nn_err_types.h"

typedef struct nn_layer_s nn_layer_t;

typedef nn_err_t (*layer_forward_fn)(matrix_t *out, const matrix_t *in,
                                     nn_layer_t *self);
typedef nn_err_t (*layer_backprop_fn)(matrix_t *grad_out, const matrix_t *grad_in,
                                      nn_layer_t *self);

struct nn_layer_s {
    const char *name;
    layer_forward_fn forward;
    layer_backprop_fn backprop;
};
#endif
