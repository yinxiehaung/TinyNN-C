#include "../include/nn.h"
#include "../include/nn_activation_layer.h"

#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#endif
#ifndef container_of
#define container_of(ptr, type, member)                                             \
    ((type *)((char *)(ptr) - offsetof(type, member)))
#endif

static nn_err_t act_layer_forward(matrix_t *out, const matrix_t *in,
                                  nn_layer_t *self)
{
    NN_PTR_NO_NULL(out, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(in, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(self, NN_ERR_NULL_PTR);
    nn_activation_layer_t *layer = container_of(self, nn_activation_layer_t, base);
    NN_REQUIRE(MAT_VALID_INTERNAL(layer->cache_in), NN_ERR_INVALID_ARG);
    out->rows = in->rows;
    out->cols = in->cols;
    mat_err_t mat_err = mat_copy(&layer->cache_in, in);
    NN_REQUIRE(mat_err == MATRIX_OK, NN_ERR_MAT_OP);
    nn_err_t err = layer->api->forward(out, in);
    NN_REQUIRE(err == NN_OK, err);
    return NN_OK;
}

static nn_err_t act_layer_backprop(matrix_t *grad_out, const matrix_t *grad_in,
                                   nn_layer_t *self)
{
    NN_PTR_NO_NULL(grad_in, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(self, NN_ERR_NULL_PTR);

    nn_activation_layer_t *layer = container_of(self, nn_activation_layer_t, base);

    if (grad_out != NULL) {
        grad_out->rows = grad_in->rows;
        grad_out->cols = grad_in->cols;
        nn_err_t err = layer->api->backprop(grad_out, grad_in, &layer->cache_in);
        NN_REQUIRE(err == NN_OK, err);
    }

    return NN_OK;
}

nn_err_t nn_model_add_activation(nn_model_t *model, const act_api_t *api)
{
    NN_PTR_MODEL_VALID(model);
    NN_PTR_NO_NULL(api, NN_ERR_NULL_PTR);
    NN_REQUIRE(model->layer_count < model->layer_capacity, NN_ERR_INVALID_MODEL);
    nn_activation_layer_t *act = NULL;
    arena_err_t a_err = arena_push(model->model_arena, sizeof(nn_activation_layer_t),
                                   true, (void **)&act);
    NN_REQUIRE(a_err == ARENA_OK, NN_ERR_INVALID_MODEL);
    act->base.name = "Activation";
    act->base.forward = act_layer_forward;
    act->base.backprop = act_layer_backprop;
    act->api = api;
    act->cache_in.data = NULL;
    act->cache_in.rows = 0;
    act->cache_in.cols = 0;
    model->layers[model->layer_count++] = (nn_layer_t *)act;
    return NN_OK;
}
