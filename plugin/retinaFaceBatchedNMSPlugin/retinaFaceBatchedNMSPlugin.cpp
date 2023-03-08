/*
 * Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "retinaFaceBatchedNMSPlugin.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::RetinaFaceBatchedNMSBasePluginCreator;
using nvinfer1::plugin::RetinaFaceBatchedNMSDynamicPlugin;
using nvinfer1::plugin::RetinaFaceBatchedNMSDynamicPluginCreator;
using nvinfer1::plugin::RetinaFaceBatchedNMSPlugin;
using nvinfer1::plugin::RetinaFaceBatchedNMSPluginCreator;
using nvinfer1::plugin::NMSParameters;

namespace
{
const char* NMS_PLUGIN_VERSION{"1"};
const char* NMS_PLUGIN_NAMES[] = {"RetinaFaceBatchedNMS_TRT", "RetinaFaceBatchedNMSDynamic_TRT"};
} // namespace

PluginFieldCollection RetinaFaceBatchedNMSBasePluginCreator::mFC{};
std::vector<PluginField> RetinaFaceBatchedNMSBasePluginCreator::mPluginAttributes;

static inline pluginStatus_t checkParams(const NMSParameters& param)
{
    // NMS plugin supports maximum thread blocksize of 512 and upto 8 blocks at once.
    constexpr int32_t maxTopK{512*8};
    if (param.topK > maxTopK)
    {
        gLogError << "Invalid parameter: NMS topK (" << param.topK << ") exceeds limit (" << maxTopK << ")" << std::endl;
        return STATUS_BAD_PARAM;
    }

    return STATUS_SUCCESS;
}

RetinaFaceBatchedNMSPlugin::RetinaFaceBatchedNMSPlugin(NMSParameters params)
    : param(params)
{
    mScoreBits = 16;
    mPluginStatus = checkParams(param);
}

RetinaFaceBatchedNMSPlugin::RetinaFaceBatchedNMSPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    ASSERT(d == a + length);

    mPluginStatus = checkParams(param);
}

RetinaFaceBatchedNMSDynamicPlugin::RetinaFaceBatchedNMSDynamicPlugin(NMSParameters params)
    : param(params)
{
    mScoreBits = 16;
    mPluginStatus = checkParams(param);
}

RetinaFaceBatchedNMSDynamicPlugin::RetinaFaceBatchedNMSDynamicPlugin(const void* data, size_t length)
{
    const char *d = reinterpret_cast<const char*>(data), *a = d;
    param = read<NMSParameters>(d);
    boxesSize = read<int>(d);
    scoresSize = read<int>(d);
    numPriors = read<int>(d);
    mClipBoxes = read<bool>(d);
    mPrecision = read<DataType>(d);
    mScoreBits = read<int32_t>(d);
    ASSERT(d == a + length);

    mPluginStatus = checkParams(param);
}

int RetinaFaceBatchedNMSPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int RetinaFaceBatchedNMSDynamicPlugin::getNbOutputs() const noexcept
{
    return 4;
}

int RetinaFaceBatchedNMSPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

int RetinaFaceBatchedNMSDynamicPlugin::initialize() noexcept
{
    return STATUS_SUCCESS;
}

void RetinaFaceBatchedNMSPlugin::terminate() noexcept {}

void RetinaFaceBatchedNMSDynamicPlugin::terminate() noexcept {}

Dims RetinaFaceBatchedNMSPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept
{
    try
    {
        ASSERT(nbInputDims == 3);
        ASSERT(index >= 0 && index < this->getNbOutputs());
        ASSERT(inputs[0].nbDims == 3);
        ASSERT(inputs[1].nbDims == 2 || (inputs[1].nbDims == 3 && inputs[1].d[2] == 1));
        ASSERT(inputs[2].nbDims == 3);
        // boxesSize: number of box coordinates for one sample
        boxesSize = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
        // scoresSize: number of scores for one sample
        scoresSize = inputs[1].d[0] * inputs[1].d[1];
        // num_detections
        if (index == 0)
        {
            Dims dim1{};
            dim1.nbDims = 1;
            dim1.d[0] = 1;
            return dim1;
        }
        // nmsed_boxes
        if (index == 1)
        {
            return DimsHW(param.keepTopK, 4);
        }
        // nmsed_scores or nmsed_classes
        if (index == 2)
        {
            Dims dim1{};
            dim1.nbDims = 1;
            dim1.d[0] = param.keepTopK;
            return dim1;
        }
        // nmsed_landmarks
        if (index==3)
        {
            return DimsHW(param.keepTopK, 10);
        }

    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return Dims{};
}

DimsExprs RetinaFaceBatchedNMSDynamicPlugin::getOutputDimensions(
    int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept
{
    try
    {
        ASSERT(nbInputs == 3);
        ASSERT(outputIndex >= 0 && outputIndex < this->getNbOutputs());

        ASSERT(inputs[0].nbDims == 4);
        ASSERT(inputs[1].nbDims == 3 || inputs[1].nbDims == 4);
        ASSERT(inputs[2].nbDims == 4);

        // set boxesSize and scoresSize
        if (inputs[0].d[0]->isConstant() && inputs[0].d[1]->isConstant() && inputs[0].d[2]->isConstant()
            && inputs[0].d[3]->isConstant())
        {
            boxesSize = exprBuilder
                            .operation(DimensionOperation::kPROD,
                                *exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[1], *inputs[0].d[2]),
                                *inputs[0].d[3])
                            ->getConstantValue();
        }

        if (inputs[1].d[0]->isConstant() && inputs[1].d[1]->isConstant() && inputs[1].d[2]->isConstant())
        {
            scoresSize = exprBuilder.operation(DimensionOperation::kPROD, *inputs[1].d[1], *inputs[1].d[2])
                             ->getConstantValue();
        }

        // cal out_dim
        DimsExprs out_dim;
        switch (outputIndex)
        {
        case 0: // nms_num_detections
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(1);
            break;
        }
        case 1: // nms_boxes
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            out_dim.d[2] = exprBuilder.constant(4);
            break;
        }
        case 2: // nms_scores
        {
            out_dim.nbDims = 2;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            break;
        }
        case 3: // nms_ldmks
        {
            out_dim.nbDims = 3;
            out_dim.d[0] = inputs[0].d[0];
            out_dim.d[1] = exprBuilder.constant(param.keepTopK);
            out_dim.d[2] = exprBuilder.constant(10);
            break;
        }
        }
        return out_dim;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return DimsExprs{};
}

size_t RetinaFaceBatchedNMSPlugin::getWorkspaceSize(int maxBatchSize) const noexcept
{
    return detectionInferenceWorkspaceSize(param.shareLocation, maxBatchSize, boxesSize, scoresSize, param.numClasses,
        numPriors, param.topK, mPrecision, mPrecision);
}

size_t RetinaFaceBatchedNMSDynamicPlugin::getWorkspaceSize(
    const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept
{
    return detectionInferenceWorkspaceSize(param.shareLocation, inputs[0].dims.d[0], boxesSize, scoresSize,
        param.numClasses, numPriors, param.topK, mPrecision, mPrecision);
}

int RetinaFaceBatchedNMSPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream) noexcept
{
    try
    {
        const void* const locData = inputs[0];
        const void* const confData = inputs[1];
        const void* const ldmkData = inputs[2];

        if (mPluginStatus != STATUS_SUCCESS)
        {
            return -1;
        }

        void* keepCount = outputs[0];
        void* nmsedBoxes = outputs[1];
        void* nmsedScores = outputs[2];
        void* nmsedLandmarks = outputs[3];

        pluginStatus_t status = retinafaceNmsInference(stream, batchSize, boxesSize, scoresSize, param.shareLocation,
            param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
            param.iouThreshold, mPrecision, locData, mPrecision, confData, ldmkData, keepCount, nmsedBoxes, nmsedScores, nmsedLandmarks,
            workspace, param.isNormalized, false, mClipBoxes, mScoreBits);
        return status == STATUS_SUCCESS ? 0 : -1;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

int RetinaFaceBatchedNMSDynamicPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc,
    const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        const void* const locData = inputs[0];
        const void* const confData = inputs[1];
        const void* const ldmkData = inputs[2];

        if (mPluginStatus != STATUS_SUCCESS)
        {
            return -1;
        }

        void* keepCount = outputs[0];
        void* nmsedBoxes = outputs[1];
        void* nmsedScores = outputs[2];
        void* nmsedLandmarks = outputs[3];

        pluginStatus_t status = retinafaceNmsInference(stream, inputDesc[0].dims.d[0], boxesSize, scoresSize, param.shareLocation,
            param.backgroundLabelId, numPriors, param.numClasses, param.topK, param.keepTopK, param.scoreThreshold,
            param.iouThreshold, mPrecision, locData, mPrecision, confData, ldmkData, keepCount, nmsedBoxes, nmsedScores, nmsedLandmarks,
            workspace, param.isNormalized, false, mClipBoxes, mScoreBits);
        return status;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return -1;
}

size_t RetinaFaceBatchedNMSPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, boxesSize,scoresSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool) + sizeof(DataType) + sizeof(int32_t);
}

void RetinaFaceBatchedNMSPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    ASSERT(d == a + getSerializationSize());
}

size_t RetinaFaceBatchedNMSDynamicPlugin::getSerializationSize() const noexcept
{
    // NMSParameters, boxesSize,scoresSize,numPriors
    return sizeof(NMSParameters) + sizeof(int) * 3 + sizeof(bool) + sizeof(DataType) + sizeof(int32_t);
}

void RetinaFaceBatchedNMSDynamicPlugin::serialize(void* buffer) const noexcept
{
    char *d = reinterpret_cast<char*>(buffer), *a = d;
    write(d, param);
    write(d, boxesSize);
    write(d, scoresSize);
    write(d, numPriors);
    write(d, mClipBoxes);
    write(d, mPrecision);
    write(d, mScoreBits);
    ASSERT(d == a + getSerializationSize());
}

void RetinaFaceBatchedNMSPlugin::configurePlugin(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs,
    const DataType* inputTypes, const DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, nvinfer1::PluginFormat format, int maxBatchSize) noexcept
{
    try
    {
        ASSERT(nbInputs == 3);
        ASSERT(nbOutputs == 4);
        ASSERT(inputDims[0].nbDims == 3);
        ASSERT(inputDims[1].nbDims == 2 || (inputDims[1].nbDims == 3 && inputDims[1].d[2] == 1));
        ASSERT(std::none_of(inputIsBroadcast, inputIsBroadcast + nbInputs, [](bool b) { return b; }));
        ASSERT(std::none_of(outputIsBroadcast, outputIsBroadcast + nbInputs, [](bool b) { return b; }));

        boxesSize = inputDims[0].d[0] * inputDims[0].d[1] * inputDims[0].d[2];
        scoresSize = inputDims[1].d[0] * inputDims[1].d[1];
        // num_boxes
        numPriors = inputDims[0].d[0];
        const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
        // Third dimension of boxes must be either 1 or num_classes
        ASSERT(inputDims[0].d[1] == numLocClasses);
        ASSERT(inputDims[0].d[2] == 4);
        mPrecision = inputTypes[0];
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

void RetinaFaceBatchedNMSDynamicPlugin::configurePlugin(
    const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept
{
    try
    {
        ASSERT(nbInputs == 3);
        ASSERT(nbOutputs == 4);

        // Shape of boxes input should be
        // Constant shape: [batch_size, num_boxes, num_classes, 4] or [batch_size, num_boxes, 1, 4]
        //           shareLocation ==              0               or          1
        const int numLocClasses = param.shareLocation ? 1 : param.numClasses;
        ASSERT(in[0].desc.dims.nbDims == 4);
        ASSERT(in[0].desc.dims.d[2] == numLocClasses);
        ASSERT(in[0].desc.dims.d[3] == 4);

        // Shape of scores input should be
        // Constant shape: [batch_size, num_boxes, num_classes] or [batch_size, num_boxes, num_classes, 1]
        ASSERT(in[1].desc.dims.nbDims == 3 || (in[1].desc.dims.nbDims == 4 && in[1].desc.dims.d[3] == 1));

        boxesSize = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        scoresSize = in[1].desc.dims.d[1] * in[1].desc.dims.d[2];
        // num_boxes
        numPriors = in[0].desc.dims.d[1];

        mPrecision = in[0].desc.type;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

bool RetinaFaceBatchedNMSPlugin::supportsFormat(DataType type, PluginFormat format) const noexcept
{
    return ((type == DataType::kHALF || type == DataType::kFLOAT || type == DataType::kINT32)
        && format == PluginFormat::kLINEAR);
}

bool RetinaFaceBatchedNMSDynamicPlugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept
{
    ASSERT(nbInputs <= 3 && nbInputs >= 0);
    ASSERT(nbOutputs <= 4 && nbOutputs >= 0);
    ASSERT(pos < 7 && pos >= 0);
    const auto* in = inOut;
    const auto* out = inOut + nbInputs;
    const bool consistentFloatPrecision = inOut[0].type == inOut[pos].type;
    switch (pos)
    {
    // inputs
    case 0:
        return (in[0].type == DataType::kHALF || in[0].type == DataType::kFLOAT)
            && in[0].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 1:
        return (in[1].type == DataType::kHALF || in[1].type == DataType::kFLOAT)
            && in[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 2:
        return (in[2].type == DataType::kHALF || in[2].type == DataType::kFLOAT)
            && in[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;

    // outputs
    case 3:
        return out[0].type == DataType::kINT32 && out[0].format == PluginFormat::kLINEAR;
    case 4:
        return (out[1].type == DataType::kHALF || out[1].type == DataType::kFLOAT)
            && out[1].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 5:
        return (out[2].type == DataType::kHALF || out[2].type == DataType::kFLOAT)
            && out[2].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    case 6:
        return (out[3].type == DataType::kHALF || out[3].type == DataType::kFLOAT)
            && out[3].format == PluginFormat::kLINEAR && consistentFloatPrecision;
    }
    return false;
}

const char* RetinaFaceBatchedNMSPlugin::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[0];
}

const char* RetinaFaceBatchedNMSDynamicPlugin::getPluginType() const noexcept
{
    return NMS_PLUGIN_NAMES[1];
}

const char* RetinaFaceBatchedNMSPlugin::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

const char* RetinaFaceBatchedNMSDynamicPlugin::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

void RetinaFaceBatchedNMSPlugin::destroy() noexcept
{
    delete this;
}

void RetinaFaceBatchedNMSDynamicPlugin::destroy() noexcept
{
    delete this;
}

IPluginV2Ext* RetinaFaceBatchedNMSPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new RetinaFaceBatchedNMSPlugin(param);
        plugin->boxesSize = boxesSize;
        plugin->scoresSize = scoresSize;
        plugin->numPriors = numPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* RetinaFaceBatchedNMSDynamicPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new RetinaFaceBatchedNMSDynamicPlugin(param);
        plugin->boxesSize = boxesSize;
        plugin->scoresSize = scoresSize;
        plugin->numPriors = numPriors;
        plugin->setPluginNamespace(mNamespace.c_str());
        plugin->setClipParam(mClipBoxes);
        plugin->mPrecision = mPrecision;
        plugin->setScoreBits(mScoreBits);
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

void RetinaFaceBatchedNMSPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* RetinaFaceBatchedNMSPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void RetinaFaceBatchedNMSDynamicPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    try
    {
        mNamespace = pluginNamespace;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
}

const char* RetinaFaceBatchedNMSDynamicPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

nvinfer1::DataType RetinaFaceBatchedNMSPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

nvinfer1::DataType RetinaFaceBatchedNMSDynamicPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept
{
    if (index == 0)
    {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

void RetinaFaceBatchedNMSPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void RetinaFaceBatchedNMSDynamicPlugin::setClipParam(bool clip) noexcept
{
    mClipBoxes = clip;
}

void RetinaFaceBatchedNMSPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

void RetinaFaceBatchedNMSDynamicPlugin::setScoreBits(int32_t scoreBits) noexcept
{
    mScoreBits = scoreBits;
}

bool RetinaFaceBatchedNMSPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const noexcept
{
    return false;
}

bool RetinaFaceBatchedNMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept
{
    return false;
}

RetinaFaceBatchedNMSBasePluginCreator::RetinaFaceBatchedNMSBasePluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("shareLocation", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("backgroundLabelId", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("numClasses", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("topK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("keepTopK", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("iouThreshold", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("isNormalized", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("clipBoxes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("scoreBits", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* RetinaFaceBatchedNMSPluginCreator::getPluginName() const noexcept
{
    return NMS_PLUGIN_NAMES[0];
}

const char* RetinaFaceBatchedNMSDynamicPluginCreator::getPluginName() const noexcept
{
    return NMS_PLUGIN_NAMES[1];
}

const char* RetinaFaceBatchedNMSBasePluginCreator::getPluginVersion() const noexcept
{
    return NMS_PLUGIN_VERSION;
}

const PluginFieldCollection* RetinaFaceBatchedNMSBasePluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV2Ext* RetinaFaceBatchedNMSPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        NMSParameters params;
        const PluginField* fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;

        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                params.shareLocation = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "topK"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreThreshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attrName, "iouThreshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "clipBoxes"))
            {
                clipBoxes = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                scoreBits = *(static_cast<const int32_t*>(fields[i].data));
            }
        }

        auto* plugin = new RetinaFaceBatchedNMSPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* RetinaFaceBatchedNMSDynamicPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc) noexcept
{
    try
    {
        NMSParameters params;
        const PluginField* fields = fc->fields;
        bool clipBoxes = true;
        int32_t scoreBits = 16;

        for (int i = 0; i < fc->nbFields; ++i)
        {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "shareLocation"))
            {
                params.shareLocation = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "backgroundLabelId"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.backgroundLabelId = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "numClasses"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.numClasses = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "topK"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.topK = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "keepTopK"))
            {
                ASSERT(fields[i].type == PluginFieldType::kINT32);
                params.keepTopK = *(static_cast<const int*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreThreshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.scoreThreshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attrName, "iouThreshold"))
            {
                ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
                params.iouThreshold = *(static_cast<const float*>(fields[i].data));
            }
            else if (!strcmp(attrName, "isNormalized"))
            {
                params.isNormalized = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "clipBoxes"))
            {
                clipBoxes = *(static_cast<const bool*>(fields[i].data));
            }
            else if (!strcmp(attrName, "scoreBits"))
            {
                scoreBits = *(static_cast<const int32_t*>(fields[i].data));
            }
        }

        auto* plugin = new RetinaFaceBatchedNMSDynamicPlugin(params);
        plugin->setClipParam(clipBoxes);
        plugin->setScoreBits(scoreBits);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2Ext* RetinaFaceBatchedNMSPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto* plugin = new RetinaFaceBatchedNMSPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}

IPluginV2DynamicExt* RetinaFaceBatchedNMSDynamicPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed, which will
        // call NMS::destroy()
        auto* plugin = new RetinaFaceBatchedNMSDynamicPlugin(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (const std::exception& e)
    {
        caughtError(e);
    }
    return nullptr;
}
