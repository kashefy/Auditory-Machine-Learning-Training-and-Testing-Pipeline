function trainAndTest_BRIR( classname )

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
pipe.modelCreator.verbose( 'on' );

pipe.trainset = 'learned_models/IdentityKS/trainTestSets/trainSet_miniMini1.flist';
pipe.testset = [];
pipe.setupData();

sc = sceneConfig.SceneConfiguration();
sc.addSource( sceneConfig.BRIRsource( ...
    brirs{1}, 'speakerId', 1 )...
    );
sc.addSource( sceneConfig.BRIRsource( brirs{1}, ...
    'data',sceneConfig.FileListValGen(pipe.pipeline.trainSet('general',:,'wavFileName')),...
    'offset', sceneConfig.ValGen('manual',0.0),...
    'speakerId', 2 ),... 
    sceneConfig.ValGen( 'manual', 10 ),...
    true );
sc.setBRIRazm( 0.4 ); % point of recorded azm range (0..1)

pipe.init( sc, 'hrir', [] );
modelPath = pipe.pipeline.run( {classname}, 0 );


pipe.modelCreator = ...
    modelTrainers.LoadModelNoopTrainer( ...
        @(cn)(fullfile( modelPath, [cn '.model.mat'] )), ...
        'performanceMeasure', @performanceMeasures.BAC2,...
        'maxDataSize', inf ...
        );

pipe.trainset = [];
pipe.testset = 'learned_models/IdentityKS/trainTestSets/testSet_miniMini1.flist';
pipe.setupData();

sc = sceneConfig.SceneConfiguration();
sc.addSource( sceneConfig.BRIRsource( ...
    brirs{1}, 'speakerId', 1 )...
    );
sc.addSource( sceneConfig.BRIRsource( brirs{1}, ...
    'data',sceneConfig.FileListValGen(pipe.pipeline.testSet('general',:,'wavFileName')),...
    'offset', sceneConfig.ValGen('manual',0.0),...
    'speakerId', 2 ),... 
    sceneConfig.ValGen( 'manual', 10 ),...
    true );
sc.setBRIRazm( 0.4 ); % point of recorded azm range (0..1)

pipe.init( sc, 'hrir', [] );
modelPath1 = pipe.pipeline.run( {classname}, 0 );

fprintf( ' Training -- Saved at %s -- \n\n', modelPath );
fprintf( ' Testing -- Saved at %s -- \n\n', modelPath1 );
