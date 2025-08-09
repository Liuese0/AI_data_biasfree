import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm, Controller } from 'react-hook-form';
import { useMutation } from '@tanstack/react-query';
import { toast } from 'react-hot-toast';
import {
  InformationCircleIcon,
  SparklesIcon,
  AdjustmentsHorizontalIcon,
} from '@heroicons/react/24/outline';
import { api } from '../services/api';
import { useStore, useSettings } from '../store';
import {
  GenerationFormData,
  DataType,
  Domain,
  OutputFormat,
  GenerationRequest,
} from '../types';
import LoadingSpinner from '../components/common/LoadingSpinner';

const Generation: React.FC = () => {
  const navigate = useNavigate();
  const { addNotification } = useStore();
  const settings = useSettings();
  const [showAdvanced, setShowAdvanced] = useState(false);

  const {
    register,
    control,
    handleSubmit,
    watch,
    formState: { errors, isSubmitting },
  } = useForm<GenerationFormData>({
    defaultValues: {
      prompt: '',
      dataType: settings.defaults.dataType,
      domain: settings.defaults.domain,
      quantity: settings.defaults.quantity,
      outputFormat: settings.defaults.outputFormat,
      enableBiasMitigation: true,
      biasMitigation: {
        demographicBalance: true,
        genderBalance: true,
        ageBalance: true,
        culturalDiversity: true,
        targetBiasReduction: 0.15,
      },
      qualityConstraints: {
        minQualityScore: 0.85,
        minSemanticSimilarity: 0.70,
        maxReidentificationRisk: 0.05,
      },
      ensemble: {
        numGenerators: 5,
        votingMethod: 'weighted_average',
        diversityWeight: 0.3,
      },
    },
  });

  const enableBiasMitigation = watch('enableBiasMitigation');
  const dataType = watch('dataType');

  const generateMutation = useMutation({
    mutationFn: (data: GenerationRequest) => api.generation.create(data),
    onSuccess: (response) => {
      toast.success('데이터 생성이 시작되었습니다');
      addNotification({
        type: 'success',
        title: '생성 시작',
        message: `작업 ID: ${response.job_id}`,
      });
      navigate(`/jobs/${response.job_id}`);
    },
    onError: (error: any) => {
      toast.error('데이터 생성 실패');
      addNotification({
        type: 'error',
        title: '생성 실패',
        message: error.message || '알 수 없는 오류가 발생했습니다',
      });
    },
  });

  const onSubmit = (data: GenerationFormData) => {
    const request: GenerationRequest = {
      prompt: data.prompt,
      data_type: data.dataType,
      domain: data.domain,
      quantity: data.quantity,
      output_format: data.outputFormat,
      bias_mitigation_config: {
        demographic_balance: data.enableBiasMitigation && data.biasMitigation.demographicBalance,
        gender_balance: data.enableBiasMitigation && data.biasMitigation.genderBalance,
        age_balance: data.enableBiasMitigation && data.biasMitigation.ageBalance,
        cultural_diversity: data.enableBiasMitigation && data.biasMitigation.culturalDiversity,
        target_bias_reduction: data.biasMitigation.targetBiasReduction,
      },
      quality_constraints: {
        min_quality_score: data.qualityConstraints.minQualityScore,
        min_semantic_similarity: data.qualityConstraints.minSemanticSimilarity,
        max_reidentification_risk: data.qualityConstraints.maxReidentificationRisk,
      },
      ensemble_config: {
        num_generators: data.ensemble.numGenerators,
        voting_method: data.ensemble.votingMethod,
        diversity_weight: data.ensemble.diversityWeight,
      },
    };

    if (data.advancedOptions) {
      request.domain_constraints = data.advancedOptions;
    }

    generateMutation.mutate(request);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">데이터 생성</h1>
        <p className="mt-1 text-sm text-gray-500">
          편향성이 제거된 고품질 합성 데이터를 생성합니다
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* 프롬프트 입력 */}
        <div className="card">
          <div className="card-body">
            <label htmlFor="prompt" className="label">
              <span className="flex items-center">
                프롬프트
                <InformationCircleIcon
                  className="ml-1 h-4 w-4 text-gray-400"
                  title="생성할 데이터에 대한 자연어 설명을 입력하세요"
                />
              </span>
            </label>
            <textarea
              id="prompt"
              {...register('prompt', {
                required: '프롬프트를 입력해주세요',
                minLength: {
                  value: 10,
                  message: '최소 10자 이상 입력해주세요',
                },
              })}
              rows={4}
              className="input"
              placeholder="예: 의료진과 환자 간의 대화 데이터 생성. 다양한 연령과 성별을 포함하고, 일반적인 질병 증상과 치료 과정을 다루는 내용으로 구성해주세요."
            />
            {errors.prompt && (
              <p className="mt-1 text-sm text-red-600">{errors.prompt.message}</p>
            )}
          </div>
        </div>

        {/* 기본 설정 */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">기본 설정</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* 데이터 타입 */}
              <div>
                <label htmlFor="dataType" className="label">
                  데이터 타입
                </label>
                <select
                  id="dataType"
                  {...register('dataType')}
                  className="input"
                >
                  <option value={DataType.TEXT}>텍스트</option>
                  <option value={DataType.IMAGE} disabled>
                    이미지 (준비중)
                  </option>
                  <option value={DataType.TABULAR} disabled>
                    테이블 (준비중)
                  </option>
                </select>
              </div>

              {/* 도메인 */}
              <div>
                <label htmlFor="domain" className="label">
                  도메인
                </label>
                <select
                  id="domain"
                  {...register('domain')}
                  className="input"
                >
                  <option value={Domain.GENERAL}>일반</option>
                  <option value={Domain.MEDICAL}>의료</option>
                  <option value={Domain.FINANCIAL}>금융</option>
                  <option value={Domain.LEGAL}>법률</option>
                </select>
              </div>

              {/* 수량 */}
              <div>
                <label htmlFor="quantity" className="label">
                  생성 수량
                </label>
                <input
                  type="number"
                  id="quantity"
                  {...register('quantity', {
                    required: '수량을 입력해주세요',
                    min: {
                      value: 1,
                      message: '최소 1개 이상 생성해야 합니다',
                    },
                    max: {
                      value: dataType === DataType.TEXT ? 100000 : 10000,
                      message: `최대 ${
                        dataType === DataType.TEXT ? '100,000' : '10,000'
                      }개까지 생성 가능합니다`,
                    },
                  })}
                  className="input"
                />
                {errors.quantity && (
                  <p className="mt-1 text-sm text-red-600">
                    {errors.quantity.message}
                  </p>
                )}
              </div>

              {/* 출력 형식 */}
              <div>
                <label htmlFor="outputFormat" className="label">
                  출력 형식
                </label>
                <select
                  id="outputFormat"
                  {...register('outputFormat')}
                  className="input"
                >
                  <option value={OutputFormat.JSON}>JSON</option>
                  <option value={OutputFormat.CSV}>CSV</option>
                  <option value={OutputFormat.TXT}>TXT</option>
                  <option value={OutputFormat.PARQUET}>Parquet</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* 편향성 완화 설정 */}
        <div className="card">
          <div className="card-body">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900">
                편향성 완화 설정
              </h3>
              <label className="flex items-center">
                <input
                  type="checkbox"
                  {...register('enableBiasMitigation')}
                  className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                />
                <span className="ml-2 text-sm text-gray-700">활성화</span>
              </label>
            </div>

            {enableBiasMitigation && (
              <div className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      {...register('biasMitigation.demographicBalance')}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm">인구통계 균형</span>
                  </label>
                  
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      {...register('biasMitigation.genderBalance')}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm">성별 균형</span>
                  </label>
                  
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      {...register('biasMitigation.ageBalance')}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm">연령 균형</span>
                  </label>
                  
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      {...register('biasMitigation.culturalDiversity')}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    <span className="ml-2 text-sm">문화 다양성</span>
                  </label>
                </div>

                <div>
                  <label className="label">
                    목표 편향성 감소율: {watch('biasMitigation.targetBiasReduction') * 100}%
                  </label>
                  <input
                    type="range"
                    {...register('biasMitigation.targetBiasReduction', {
                      valueAsNumber: true,
                    })}
                    min="0"
                    max="0.5"
                    step="0.05"
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>0%</span>
                    <span>25%</span>
                    <span>50%</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 품질 제약사항 */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              품질 제약사항
            </h3>
            
            <div className="space-y-4">
              <div>
                <label className="label">
                  최소 품질 점수: {watch('qualityConstraints.minQualityScore') * 100}%
                </label>
                <input
                  type="range"
                  {...register('qualityConstraints.minQualityScore', {
                    valueAsNumber: true,
                  })}
                  min="0.5"
                  max="1"
                  step="0.05"
                  className="w-full"
                />
              </div>

              <div>
                <label className="label">
                  최소 의미적 유사성: {watch('qualityConstraints.minSemanticSimilarity') * 100}%
                </label>
                <input
                  type="range"
                  {...register('qualityConstraints.minSemanticSimilarity', {
                    valueAsNumber: true,
                  })}
                  min="0.5"
                  max="1"
                  step="0.05"
                  className="w-full"
                />
              </div>

              <div>
                <label className="label">
                  최대 재식별 위험도: {watch('qualityConstraints.maxReidentificationRisk') * 100}%
                </label>
                <input
                  type="range"
                  {...register('qualityConstraints.maxReidentificationRisk', {
                    valueAsNumber: true,
                  })}
                  min="0"
                  max="0.2"
                  step="0.01"
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        {/* 앙상블 설정 */}
        <div className="card">
          <div className="card-body">
            <h3 className="text-lg font-medium text-gray-900 mb-4">
              앙상블 설정
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="label">
                  생성기 수: {watch('ensemble.numGenerators')}
                </label>
                <input
                  type="range"
                  {...register('ensemble.numGenerators', {
                    valueAsNumber: true,
                  })}
                  min="3"
                  max="10"
                  step="1"
                  className="w-full"
                />
              </div>

              <div>
                <label htmlFor="votingMethod" className="label">
                  투표 방식
                </label>
                <select
                  id="votingMethod"
                  {...register('ensemble.votingMethod')}
                  className="input"
                >
                  <option value="weighted_average">가중 평균</option>
                  <option value="majority">다수결</option>
                  <option value="unanimous">만장일치</option>
                </select>
              </div>

              <div>
                <label className="label">
                  다양성 가중치: {watch('ensemble.diversityWeight')}
                </label>
                <input
                  type="range"
                  {...register('ensemble.diversityWeight', {
                    valueAsNumber: true,
                  })}
                  min="0"
                  max="1"
                  step="0.1"
                  className="w-full"
                />
              </div>
            </div>
          </div>
        </div>

        {/* 고급 설정 */}
        <div className="card">
          <div className="card-body">
            <button
              type="button"
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="flex items-center text-sm text-gray-600 hover:text-gray-900"
            >
              <AdjustmentsHorizontalIcon className="h-5 w-5 mr-1" />
              고급 설정 {showAdvanced ? '숨기기' : '표시'}
            </button>

            {showAdvanced && (
              <div className="mt-4 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="label">온도 (Temperature)</label>
                    <input
                      type="number"
                      {...register('advancedOptions.temperature', {
                        valueAsNumber: true,
                      })}
                      min="0"
                      max="2"
                      step="0.1"
                      defaultValue={0.8}
                      className="input"
                    />
                  </div>

                  <div>
                    <label className="label">최대 길이</label>
                    <input
                      type="number"
                      {...register('advancedOptions.maxLength', {
                        valueAsNumber: true,
                      })}
                      min="10"
                      max="1000"
                      defaultValue={200}
                      className="input"
                    />
                  </div>

                  <div>
                    <label className="label">Top-P</label>
                    <input
                      type="number"
                      {...register('advancedOptions.topP', {
                        valueAsNumber: true,
                      })}
                      min="0"
                      max="1"
                      step="0.1"
                      defaultValue={0.9}
                      className="input"
                    />
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* 제출 버튼 */}
        <div className="flex justify-end space-x-4">
          <button
            type="button"
            onClick={() => navigate('/dashboard')}
            className="btn btn-outline"
            disabled={isSubmitting}
          >
            취소
          </button>
          <button
            type="submit"
            className="btn btn-primary flex items-center"
            disabled={isSubmitting}
          >
            {isSubmitting ? (
              <>
                <LoadingSpinner size="small" color="white" />
                <span className="ml-2">생성 중...</span>
              </>
            ) : (
              <>
                <SparklesIcon className="h-5 w-5 mr-2" />
                데이터 생성
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default Generation;