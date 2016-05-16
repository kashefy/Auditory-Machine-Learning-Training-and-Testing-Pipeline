classdef AzmLabeler < LabelCreators.EnergyDependentLabeler
    % class for labeling blocks by azm of a specified source
    %% -----------------------------------------------------------------------------------
    properties (SetAccess = private)
    end
    
    %% -----------------------------------------------------------------------------------
    methods (Abstract)
    end

    %% -----------------------------------------------------------------------------------
    methods
        
        function obj = AzmLabeler( varargin )
            ip = inputParser;
            ip.addOptional( 'sourceMinEnergy', -20 );
            ip.addOptional( 'labelBlockSize_s', [] );
            ip.addOptional( 'sourceId', 1 );
            ip.parse( varargin{:} );
            obj = obj@LabelCreators.EnergyDependentLabeler( ...
                                      'labelBlockSize_s', ip.Results.labelBlockSize_s, ...
                                      'sourcesMinEnergy', ip.Results.sourceMinEnergy, ...
                                      'sourceIds', ip.Results.sourceId );
        end
        %% -------------------------------------------------------------------------------

    end
    
    %% -----------------------------------------------------------------------------------
    methods (Access = protected)
        
        function outputDeps = getLabelInternOutputDependencies( obj )
            outputDeps.v = 1;
        end
        %% -------------------------------------------------------------------------------

        function y = labelEnergeticBlock( obj, blockAnnotations )
            blockAzms = [blockAnnotations.srcAzms.srcAzms{:}];
            srcBlockAzms = blockAzms(obj.sourcesId,:);
            y = median( srcBlockAzms );
        end
        %% -------------------------------------------------------------------------------
                
    end
    %% -----------------------------------------------------------------------------------
    
    methods (Static)
        
        %% -------------------------------------------------------------------------------
        
    end
    
end

        
