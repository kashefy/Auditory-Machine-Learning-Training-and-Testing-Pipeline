function buildOnlyCache( )

addPathsIfNotIncluded( cleanPathFromRelativeRefs( [pwd '/..'] ) ); 
startAMLTTP();

pipe = TwoEarsIdTrainPipe();
pipe.featureCreator = FeatureCreators.FeatureSetRmBlockmean();
babyLabeler = LabelCreators.MultiEventTypeLabeler( 'types', {{'baby'}}, 'negOut', 'rest' );
pipe.labelCreator = babyLabeler;
pipe.modelCreator = ModelTrainers.LoadModelNoopTrainer( 'noop' );

pipe.data = 'learned_models/IdentityKS/trainTestSets/NIGENS160807_mini_TrainSet_1.flist';
pipe.setupData();

sc = SceneConfig.SceneConfiguration();
sc.addSource( SceneConfig.PointSource( ...
        'data', SceneConfig.FileListValGen( 'pipeInput' )  )  );
sc.addSource( SceneConfig.PointSource( ...
        'data', SceneConfig.FileListValGen( ...
                  pipe.pipeline.data('fileLabel',{{'type',{'fire'}}},'fileName') ) ),...
    'loop', 'randomSeq',...
    'snr', SceneConfig.ValGen( 'manual', 0 ),...
    'snrRef', 1 );
pipe.init( sc, 'stopAfterProc', inf );

modelPath = pipe.pipeline.run( 'runOption', 'onlyGenCache' );

fprintf( ' -- run log is saved at %s -- \n\n', modelPath );
