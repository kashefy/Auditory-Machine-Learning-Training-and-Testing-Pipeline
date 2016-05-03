function [modelPath_train, modelPath_test] = trainAndTest_BRIR_toulouse_mc( classname )

if nargin < 1, classname = 'baby'; end;

%startTwoEars( '../IdentificationTraining.xml' );
addpath( '..' );
startIdentificationTraining();

brirs = { ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos1.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos2.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos3.sofa'; ...
    'impulse_responses/twoears_kemar_adream/TWOEARS_KEMAR_ADREAM_pos4.sofa'; ...
    };

pipe = TwoEarsIdTrainPipe();
pipe.featureCreator = featureCreators.FeatureSet1Blockmean();
pipe.modelCreator = modelTrainers.GlmNetLambdaSelectTrainer( ...
   'performanceMeasure', @performanceMeasures.BAC2, ...
   'cvFolds', 4, ...
   'alpha', 0.99 );
% pipe.modelCreator = modelTrainers.SVMmodelSelectTrainer( ...
% 'performanceMeasure', @performanceMeasures.BAC2, ...
% 'hpsEpsilons', [0.001], ... % define hps set (not a range)
% 'hpsKernels', [0], ...      % define hps set (not a range). 0 = linear, 2 = rbf
% 'hpsCrange', [-6 2], ...    % define hps C range -- logspaced between 10^a and 10^b
% 'hpsGammaRange', [-12 3], ... % define hps Gamma range -- logspaced between 10^a and 
%                           ... % 10^b. Ignored for kernel other than rbf
% 'hpsMaxDataSize', 50, ...  % max data set size to use in hps (number of samples)
% 'hpsRefineStages', 1, ...   % number of iterative hps refinement stages
% 'hpsSearchBudget', 7, ...   % number of hps grid search parameter values per dimension
% 'hpsCvFolds', 4,...         % number of hps cv folds of training set
% 'finalMaxDataSize',111);
% pipe.modelCreator.verbose( 'on' );

pipe.trainset = 'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TrainSet_1.flist';
pipe.testset = [];
pipe.setupData();

% loop over position combinatation
% once without distractor
brir_azm = 0.4; % point of recorded azm range (0..1)
sc_idx = 1;
for pos_target = 1 : numel(brirs)

    sc(sc_idx) = sceneConfig.SceneConfiguration();
    sc(sc_idx).addSource( sceneConfig.BRIRsource( ...
        brirs{pos_target}, 'speakerId', 1 )...
        );
    sc(sc_idx).setBRIRazm( brir_azm ); % point of recorded azm range (0..1)
    sc_idx = sc_idx + 1;
    
    for pos_distractor = 1 : numel(brirs)

        sc(sc_idx) = sceneConfig.SceneConfiguration();
        sc(sc_idx).addSource( sceneConfig.BRIRsource( ...
            brirs{pos_target}, 'speakerId', 1 )...
            );
        sc(sc_idx).addSource( sceneConfig.BRIRsource( brirs{pos_distractor}, ...
            'data',sceneConfig.FileListValGen(pipe.pipeline.trainSet('general',:,'wavFileName')),...
            'offset', sceneConfig.ValGen('manual',0.0),...
            'speakerId', 2 ),... 
            sceneConfig.ValGen( 'manual', 0 ),...
            true );
        sc(sc_idx).setBRIRazm( brir_azm ); % point of recorded azm range (0..1)
        sc_idx = sc_idx + 1;
    end
end
pipe.init( sc, 'hrir', [] );
modelPath_train = pipe.pipeline.run( {classname}, 0 );

pipe.modelCreator = ...
    modelTrainers.LoadModelNoopTrainer( ...
        @(cn)(fullfile( modelPath_train, [cn '.model.mat'] )), ...
        'performanceMeasure', @performanceMeasures.BAC2,...
        'maxDataSize', inf ...
        );

pipe.trainset = [];
pipe.testset = 'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TestSet_1.flist';
pipe.setupData();

pipe.init( sc, 'hrir', [] );
modelPath_test = pipe.pipeline.run( {classname}, 0 );

fprintf( ' Training -- Saved at %s -- \n\n', modelPath_train );
fprintf( ' Testing -- Saved at %s -- \n\n', modelPath_test );
