#include <stdio.h>
#include <string.h>
#include "../include/nn.h"
#include "../include/nn_dense_layer.h"
#include "../include/nn_activation_layer.h"

nn_err_t nn_model_save(const nn_model_t *model, const char *filepath)
{
    NN_PTR_NO_NULL(model, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(filepath, NN_ERR_NULL_PTR);

    FILE *fp = fopen(filepath, "wb");
    NN_REQUIRE(fp != NULL, NN_ERR_IO);

    if (fwrite(&model->layer_count, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NN_ERR_IO;
    }

    for (uint64_t i = 0; i < model->layer_count; i++) {
        nn_layer_t *layer = model->layers[i];
        uint32_t layer_type = 0;

        if (strcmp(layer->name, "Dense") == 0) {
            layer_type = 1;
            if (fwrite(&layer_type, sizeof(uint32_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }

            nn_dense_layer_t *dense = (nn_dense_layer_t *)layer;

            fwrite(&dense->w.rows, sizeof(uint64_t), 1, fp);
            fwrite(&dense->w.cols, sizeof(uint64_t), 1, fp);
            if (fwrite(dense->w.data, sizeof(mat_data_t),
                       dense->w.rows * dense->w.cols,
                       fp) != (size_t)(dense->w.rows * dense->w.cols)) {
                fclose(fp);
                return NN_ERR_IO;
            }

            fwrite(&dense->b.rows, sizeof(uint64_t), 1, fp);
            fwrite(&dense->b.cols, sizeof(uint64_t), 1, fp);
            if (fwrite(dense->b.data, sizeof(mat_data_t),
                       dense->b.rows * dense->b.cols,
                       fp) != (size_t)(dense->b.rows * dense->b.cols)) {
                fclose(fp);
                return NN_ERR_IO;
            }
        } else if (strcmp(layer->name, "Activation") == 0) {
            layer_type = 2;
            if (fwrite(&layer_type, sizeof(uint32_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }

            nn_activation_layer_t *act = (nn_activation_layer_t *)layer;
            uint32_t act_type = 0;
            if (act->api == &ACT_RELU)
                act_type = 1;
            else if (act->api == &ACT_SIGMOID)
                act_type = 2;
            else if (act->api == &ACT_SOFTMAX)
                act_type = 3;
            else if (act->api == &ACT_LINEAR)
                act_type = 4;

            if (fwrite(&act_type, sizeof(uint32_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }
        }
    }

    fclose(fp);
    return NN_OK;
}

nn_err_t nn_model_load(nn_model_t *model, const char *filepath)
{
    NN_PTR_NO_NULL(model, NN_ERR_NULL_PTR);
    NN_PTR_NO_NULL(filepath, NN_ERR_NULL_PTR);

    FILE *fp = fopen(filepath, "rb");
    NN_REQUIRE(fp != NULL, NN_ERR_IO);

    uint64_t incoming_layer_count = 0;
    if (fread(&incoming_layer_count, sizeof(uint64_t), 1, fp) != 1) {
        fclose(fp);
        return NN_ERR_IO;
    }

    NN_REQUIRE(incoming_layer_count <= model->layer_capacity, NN_ERR_INVALID_MODEL);

    const act_api_t *api = NULL;
    for (uint64_t i = 0; i < incoming_layer_count; i++) {
        uint32_t layer_type = 0;
        if (fread(&layer_type, sizeof(uint32_t), 1, fp) != 1) {
            fclose(fp);
            return NN_ERR_IO;
        }

        if (layer_type == 1) {
            uint64_t w_rows = 0, w_cols = 0;
            if (fread(&w_rows, sizeof(uint64_t), 1, fp) != 1 ||
                fread(&w_cols, sizeof(uint64_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }

            nn_err_t err = nn_model_add_dense(model, w_rows, w_cols);
            if (err != NN_OK) {
                fclose(fp);
                return err;
            }

            nn_dense_layer_t *dense =
                (nn_dense_layer_t *)model->layers[model->layer_count - 1];
            if (fread(dense->w.data, sizeof(mat_data_t), w_rows * w_cols, fp) !=
                (size_t)(w_rows * w_cols)) {
                fclose(fp);
                return NN_ERR_IO;
            }

            uint64_t b_rows = 0, b_cols = 0;
            if (fread(&b_rows, sizeof(uint64_t), 1, fp) != 1 ||
                fread(&b_cols, sizeof(uint64_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }

            if (fread(dense->b.data, sizeof(mat_data_t), b_rows * b_cols, fp) !=
                (size_t)(b_rows * b_cols)) {
                fclose(fp);
                return NN_ERR_IO;
            }
        } else if (layer_type == 2) {
            uint32_t act_type = 0;
            if (fread(&act_type, sizeof(uint32_t), 1, fp) != 1) {
                fclose(fp);
                return NN_ERR_IO;
            }
            if (act_type == 1)
                api = &ACT_RELU;
            else if (act_type == 2)
                api = &ACT_SIGMOID;
            else if (act_type == 3)
                api = &ACT_SOFTMAX;
            else if (act_type == 4)
                api = &ACT_LINEAR;
            else {
                fclose(fp);
                return NN_ERR_INVALID_MODEL;
            }

            nn_err_t err = nn_model_add_activation(model, api);
            if (err != NN_OK) {
                fclose(fp);
                return err;
            }
        }
    }

    fclose(fp);
    return NN_OK;
}
