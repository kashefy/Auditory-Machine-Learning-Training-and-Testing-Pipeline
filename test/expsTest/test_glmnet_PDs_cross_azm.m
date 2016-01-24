function test_glmnet_PDs_cross_azm(ddt,ss)
    
addpath( '../..' );
startIdentificationTraining();

featureCreators = {?featureCreators.FeatureSet1Blockmean,...
                   ?featureCreators.FeatureSet1Blockmean2Ch};
azimuths = {{0,0},... %1
    {0,45},{45,0},{22.5,-22.5},{67.5,112.5},{-157.5,157.5},... %2-6
    {0,90},{22.5,112.5},{45,135},{90,180},{22.5,-67.5},{45,-45},{90,0},{-157.5,112.5},... %7-14
    {0,180},{22.5,-157.5},{45,-135},{67.5,-112.5},{90,-90}}; % 15-19
snrs = {0,-10,10,-20};
datasets = {'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TrainSet_1.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TestSet_1.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TrainSet_2.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TestSet_2.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TrainSet_3.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TestSet_3.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TrainSet_4.flist',...
            'learned_models/IdentityKS/trainTestSets/NIGENS_75pTrain_TestSet_4.flist'
            };
classes = {'alarm','baby','femaleSpeech','fire','crash','dog','engine','footsteps',...
           'knock','phone','piano'};
crossAzms(1,:) = [1,01,01,4,04,04,12,12,12,19,19,19];
crossAzms(2,:) = [4,12,19,1,12,19,01,04,19,01,04,12];
crossAzms(1,:) = [crossAzms(1,:),ones(1,18)];
crossAzms(2,:) = [crossAzms(2,:),2:19];
       
if exist( ['pds_glmnet_test_crossAzm' strrep(num2str([ddt,ss]),' ','_') '.mat'], 'file' )
    load( ['pds_glmnet_test_crossAzm' strrep(num2str([ddt,ss]),' ','_') '.mat'] );
else
    doneCfgsTest = {};
end


for aac = 1 : size( crossAzms, 2 )
%for ddt = [2 4 6]
for cc = 1 : 4 %numel( classes )
for ff = 1 : numel( featureCreators )
aa = crossAzms(1,aac);
aat = crossAzms(2,aac);

sst = ss;
dd = ddt-1;

fprintf( '\n\n==============\nTesting %s, dd = %d, ff = %d\nss = %d, sst = %d, aa = %d, aat = %d.==============\n\n', ...
    classes{cc}, ddt, ff, ss, sst, aa, aat );

if exist( ['pds_' strrep(num2str(aa),' ','_') '_glmnet.mat'], 'file' )
    load( ['pds_' strrep(num2str(aa),' ','_') '_glmnet'] );
else
    warning( 'training mat file not found' );
    pause;
    continue;
end

if ~any( cellfun( @(x)(all(x==[cc dd ss ff aa])), doneCfgs ) )
    continue; % training not done yet
end
if any( cellfun( @(x)(all(x==[cc ddt ss ff aa sst aat])), doneCfgsTest ) )
    continue; % testing already done
end
    
pipe = TwoEarsIdTrainPipe();
pipe.featureCreator = feval( featureCreators{ff}.Name );
pipe.modelCreator = ...
    modelTrainers.LoadModelNoopTrainer( ...
        @(cn)(fullfile( modelpathes{cc,dd,ss,ff,aa}, [cn '.model.mat'] )), ...
        'performanceMeasure', @performanceMeasures.BAC2, ...
        'modelParams', struct('lambda', []) );
pipe.modelCreator.verbose( 'on' );

pipe.trainset = [];
pipe.testset = datasets{ddt};
pipe.setupData();

sc = sceneConfig.SceneConfiguration();
sc.addSource( sceneConfig.PointSource( ...
    'azimuth',sceneConfig.ValGen('manual',azimuths{aat}{1}) ) );
sc.addSource( sceneConfig.PointSource( ...
    'azimuth',sceneConfig.ValGen('manual',azimuths{aat}{2}), ...
    'data',sceneConfig.FileListValGen(pipe.pipeline.data('general',:,'wavFileName')),...
    'offset', sceneConfig.ValGen('manual',0.0) ),...
    sceneConfig.ValGen( 'manual', snrs{sst} ),...
    true ); % loop
pipe.setSceneConfig( sc );

pipe.init();
modelpathes_test{cc,ddt,ss,ff,aa,sst,aat} = pipe.pipeline.run( classes(cc), 0 );

testmodel = load( [modelpathes_test{cc,ddt,ss,ff,aa,sst,aat} filesep classes{cc} '.model.mat'] );

test_performances{cc,ddt,ss,ff,aa,sst,aat} = [testmodel.testPerfresults.performance];
[~,~,~,lambda_b{cc,ddt,ss,ff,aa,sst,aat},~] = ...
    testmodel.model.getBestLambdaCVresults();
[~,~,~,lambda_hws{cc,ddt,ss,ff,aa,sst,aat},~] = ...
    testmodel.model.getHighestLambdaWithinStdCVresults();
lbIdx = find( testmodel.model.model.lambda == lambda_b{cc,ddt,ss,ff,aa,sst,aat} );
lhwsIdx = find( testmodel.model.model.lambda == lambda_hws{cc,ddt,ss,ff,aa,sst,aat} );
test_performances_b{cc,ddt,ss,ff,aa,sst,aat} = test_performances{cc,ddt,ss,ff,aa,sst,aat}(lbIdx);
test_performances_hws{cc,ddt,ss,ff,aa,sst,aat} = test_performances{cc,ddt,ss,ff,aa,sst,aat}(lhwsIdx);

doneCfgsTest{end+1} = [cc ddt ss ff aa sst aat];

save( ['pds_glmnet_test_crossAzm' strrep(num2str([ddt,ss]),' ','_') '.mat'], ...
    'modelpathes_test', 'doneCfgsTest', ...
    'test_performances', ...
    'lambda_b', ...
    'lambda_hws', ...
    'test_performances_b', 'test_performances_hws' );

end
end
%end
%end
end

