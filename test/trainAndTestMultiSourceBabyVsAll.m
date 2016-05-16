function trainAndTestMultiSourceBabyVsAll()

addPathsIfNotIncluded( cleanPathFromRelativeRefs( [pwd '/..'] ) ); 
startIdentificationTraining();

pipe = TwoEarsIdTrainPipe();
pipe.blockCreator = BlockCreators.DistractedBlockCreator( 1.0, 0.4, ...
                                                          'distractorSources', [2 3],...
                                                          'rejectEnergyThreshold', -30 );
pipe.featureCreator = FeatureCreators.FeatureSet1Blockmean();
babyLabeler = LabelCreators.BinaryEventTypeLabeler( 'posOutType', {'baby'} );
pipe.labelCreator = babyLabeler;
%% other labeler examples
% 
% % baby will be 1, fire -1, labels based on only 0.2s
% babyVsFireLabeler = LabelCreators.BinaryEventTypeLabeler( 'posOutType', {'baby'}, ...
%                                                           'negOut', 'event', ...
%                                                           'negOutType', {'fire'}, ...
%                                                           'labelBlockSize_s', 0.2 );
% pipe.labelCreator = babyVsFireLabeler;
% 
% % baby+female will be 1, rest -1
% babyFemaleVsRestLabeler = ... 
%     LabelCreators.BinaryEventTypeLabeler( 'posOutType', {'baby', 'femaleSpeech'}, ...
%                                           'negOut', 'all', 'negOutType', 'rest' );
% pipe.labelCreator = babyFemaleVsRestLabeler;
% 
% % alarm will be 1, baby 2, female 3, fire 4
% typeMulticlassLabeler = ... 
%     LabelCreators.MultiEventTypeLabeler( 'types', ...
%                                        {{'alarm'},{'baby'},{'femaleSpeech'},{'fire'}} );
% pipe.labelCreator = typeMulticlassLabeler;
% 
% % multinomial labels: baby will be (1,1,2), fire (-1,-1,4), female (-1,0,3)
% multiLabeler = LabelCreators.MultiLabeler( ...
%                                   {babyLabeler,babyVsFireLabeler,typeMulticlassLabeler} );
% pipe.labelCreator = multiLabeler;
% 
% % label will be azm of source 1
% azmLabeler = LabelCreators.AzmLabeler( 'sourceId', 1 );
% pipe.labelCreator = azmLabeler;
% 
% % label will be distribution of sources over azm
% azmLabeler2 = LabelCreators.AzmDistributionLabeler( 'sourcesMinEnergy', -30 );
% pipe.labelCreator = azmLabeler2;
% 
% % label will be number of sources
% noSrcsLabeler = LabelCreators.NumberOfSourcesLabeler();
% pipe.labelCreator = noSrcsLabeler;
% 
% % multinomial labels: (typeId,azmSrc1)
% multiLabeler = LabelCreators.MultiLabeler( {typeMulticlassLabeler, azmLabeler} );
% pipe.labelCreator = multiLabeler;
%
%%
pipe.modelCreator = ModelTrainers.GlmNetLambdaSelectTrainer( ...
    'performanceMeasure', @PerformanceMeasures.BAC2, ...
    'cvFolds', 4, ...
    'alpha', 0.99 );
pipe.modelCreator.verbose( 'on' );

pipe.trainset = 'learned_models\IdentityKS\trainTestSets/NIGENS_10pTrain_TrainSet_1.flist';
pipe.setupData();

sc = SceneConfig.SceneConfiguration();
sc.addSource( SceneConfig.PointSource( ...
        'data', SceneConfig.FileListValGen( 'pipeInput' ),...
        'normalize', true, 'normalizeLevel', 0.9 ),...
    'snr', SceneConfig.ValGen( 'manual', 0 ),...
    'snrRef', 1 );
sc.addSource( SceneConfig.PointSource( ...
        'data', SceneConfig.FileListValGen( ...
                  pipe.pipeline.trainSet('fileLabel',{{'type',{'fire'}}},'fileName') ),...
        'normalize', false ),...
    'loop', 'no',...
    'snr', SceneConfig.ValGen( 'manual', 0 ),...
    'snrRef', 1 );
sc.addSource( SceneConfig.PointSource( ...
        'data', SceneConfig.FileListValGen( ...
               pipe.pipeline.trainSet('fileLabel',{{'type',{'general'}}},'fileName') ),...
        'offset', SceneConfig.ValGen( 'manual', 0 ) ),...
    'snr', SceneConfig.ValGen( 'manual', 10 ),...
    'loop', 'randomSeq' );
sc.setLengthRef( 'source', 1, 'min', 30 );
pipe.init( sc );

modelPath = pipe.pipeline.run( 'interestingModel' );

fprintf( ' -- Model is saved at %s -- \n\n', modelPath );

% TODO: test