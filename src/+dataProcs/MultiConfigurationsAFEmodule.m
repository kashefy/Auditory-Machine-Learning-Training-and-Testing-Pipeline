classdef MultiConfigurationsAFEmodule < core.IdProcInterface
    
    %% --------------------------------------------------------------------
    properties (SetAccess = private)
        afeProc;
        singleConfFiles;
        singleConfs;
        outputWavFileName;
    end
    
    %% --------------------------------------------------------------------
    methods (Static)
    end
    
    %% --------------------------------------------------------------------
    methods (Access = public)
        
        function obj = MultiConfigurationsAFEmodule( afeProc )
            obj = obj@core.IdProcInterface();
            if ~isa( afeProc, 'core.IdProcInterface' )
                error( 'afeProc must implement core.IdProcInterface.' );
            end
            obj.afeProc = afeProc;
        end
        %% ----------------------------------------------------------------

        function process( obj, inputFileName )
            obj.makeAFEdata( inputFileName );
        end
        
    end

    %% --------------------------------------------------------------------
    methods (Access = protected)
        
        function outputDeps = getInternOutputDependencies( obj )
            outputDeps.afeDeps = obj.afeProc.getInternOutputDependencies;
        end
        %% ----------------------------------------------------------------

        function out = getOutput( obj )
            out.singleConfFiles = obj.singleConfFiles;
            out.singleConfs = obj.singleConfs;
            out.wavFileName = obj.outputWavFileName;
        end
        %% ----------------------------------------------------------------
        
        function makeAFEdata( obj, inFileName )
            [p,wavFileName,~] = fileparts( inFileName );
            [~,wavFileName,~] = fileparts( wavFileName );
            soundDir = fileparts( p );
            wavFileName = fullfile( soundDir, wavFileName );
            obj.outputWavFileName = wavFileName;
            obj.singleConfFiles = {};
            obj.singleConfs = [];
            multiCfg = obj.getOutputDependencies();
            scFieldNames = fieldnames( multiCfg.extern );
            for ii = 1 : numel( scFieldNames )
                conf = multiCfg.extern.(scFieldNames{ii});
                obj.afeProc.setExternOutputDependencies( conf );
                if ~obj.afeProc.hasFileAlreadyBeenProcessed( wavFileName )
                    in = load( inFileName );
                    if ~exist( in.singleScFiles{ii}, 'file' )
                        error( '%s not found. \n%s corrupt -- delete and restart.', ...
                            in.singleScFiles{ii}, inFileName );
                    end
                    obj.afeProc.process( in.singleScFiles{ii} );
                    obj.afeProc.saveOutput( wavFileName );
                end
                obj.singleConfFiles{ii} = obj.afeProc.getOutputFileName( wavFileName );
                obj.singleConfs{ii} = obj.afeProc.getOutputDependencies;
                fprintf( ';' );
            end
            fprintf( '\n' );
        end
        %% ----------------------------------------------------------------
        
        function precProcFileNeeded = needsPrecedingProcResult( obj, wavFileName )
            precProcFileNeeded = false; 
            multiCfg = obj.getOutputDependencies();
            scFieldNames = fieldnames( multiCfg.extern );
            for ii = 1 : numel( scFieldNames )
                conf = multiCfg.extern.(scFieldNames{ii});
                obj.afeProc.setExternOutputDependencies( conf );
                if ~obj.afeProc.hasFileAlreadyBeenProcessed( wavFileName )
                    precProcFileNeeded = true;
                    return;
                end
            end
        end
        %% -----------------------------------------------------------------
    end
    
    %% --------------------------------------------------------------------
    methods (Access = private)
    end
    
end
