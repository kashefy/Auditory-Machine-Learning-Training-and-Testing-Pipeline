classdef SVMmodel < IdModelInterface
    
    %% --------------------------------------------------------------------
    properties (SetAccess = protected)
        dataTranslators;
        dataScalors;
    end
    
    %% --------------------------------------------------------------------
    properties (SetAccess = public)
        useProbModel;
        model;
    end
    
    %% --------------------------------------------------------------------
    methods

        function obj = SVMmodel()
            obj.dataTranslators = 0;
            obj.dataScalors = 1;
        end
        %% -----------------------------------------------------------------
        
        function [y,score] = applyModel( obj, x )
            x = obj.scale2zeroMeanUnitVar( x, false );
            yDummy = zeros( size( x, 1 ), 1 );
            [y, ~, score] = libsvmpredict( yDummy, x, obj.model, sprintf( '-q -b %d', obj.useProbModel ) );
        end
        %% -----------------------------------------------------------------
        
        function x = scale2zeroMeanUnitVar( obj, x, saveScalingFactors )
            if isempty( x ), return; end;
            if saveScalingFactors
                obj.dataTranslators = mean( x );
                obj.dataScalors = 1 ./ std( x );
            end
            x = x - repmat( obj.dataTranslators, size(x,1), 1 );
            x = x .* repmat( obj.dataScalors, size(x,1), 1 );
        end
        %% -----------------------------------------------------------------
        
    end
    
end

