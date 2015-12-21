function trainAndTestMultiSource( classname )

if nargin < 1, classname = 'baby'; end;

%startTwoEars( '../IdentificationTraining.xml' );
addpath( '..' );
startIdentificationTraining();

pipe = TwoEarsIdTrainPipe();
pipe.featureCreator = featureCreators.FeatureSet1Blockmean();
pipe.blockCreator = blockCreators.DistractedBlockCreator( [2 3] );
pipe.blockCreator.labelCreator = blockCreators.ObjectTypeLabeler( 'binary', 1 );
pipe.modelCreator = modelTrainers.GlmNetLambdaSelectTrainer( ...
    'performanceMeasure', @performanceMeasures.BAC2, ...
    'cvFolds', 4, ...
    'alpha', 0.99 );
pipe.modelCreator.verbose( 'on' );

pipe.trainset = 'testFlists/NIGENS_trainSet_miniMini.flist';
pipe.setupData();

sc = sceneConfig.SceneConfiguration();
sc.addSource( sceneConfig.PointSource( ...
        'data', sceneConfig.FileListValGen( pipe.pipeline.trainSet(:,:,'wavFileName'), 'all' ),...
        'normalize', true,...
        'loop', 'none' ),...
    'snr', sceneConfig.ValGen( 'manual', 0 ),...
    'snrRef', 1 );
sc.addSource( sceneConfig.PointSource( ...
        'data', sceneConfig.FileListValGen( pipe.pipeline.trainSet('fire',:,'wavFileName'), 'all' ),...
        'normalize', true,...
        'loop', 'none' ),...
    'snr', sceneConfig.ValGen( 'manual', 0 ),...
    'snrRef', 1 );
sc.addSource( sceneConfig.PointSource( ...
        'data', sceneConfig.FileListValGen( pipe.pipeline.trainSet('general',:,'wavFileName'), 'one' ),...
        'normalize', true,...
        'loop', 'randomSeq' ),...
    'snr', sceneConfig.ValGen( 'manual', 10 ),...
    'snrRef', 1 );
sc.setLengthRef( 'source', 1 );
pipe.setSceneConfig( [sc] ); 

pipe.init();
modelPath = pipe.pipeline.run( {classname}, 0 );

fprintf( ' -- Model is saved at %s -- \n\n', modelPath );

pipe.modelCreator = ...
    modelTrainers.LoadModelNoopTrainer( ...
        @(cn)(fullfile( modelPath, [cn '.model.mat'] )), ...
        'performanceMeasure', @performanceMeasures.BAC2,...
        'maxDataSize', inf ...
        );

pipe.trainset = [];
pipe.testset = 'testFlists/NIGENS_testSet_miniMini.flist';
pipe.setupData();

% TODO adapt
%
% sc = sceneConfig.SceneConfiguration(); % clean
% sc.addSource( sceneConfig.PointSource() );
% sc.addSource( sceneConfig.PointSource( ...
%     'data',sceneConfig.FileListValGen(pipe.pipeline.testSet('general',:,'wavFileName')) ),...
%     sceneConfig.ValGen( 'manual', 10 ));
% pipe.setSceneConfig( [sc] ); 

pipe.init();
modelPath = pipe.pipeline.run( {classname}, 0 );
