classdef FeatureSetNSrcDetection < FeatureCreators.Base
% FeatureSetNSrc   Number of Sources estimation with features consisting of:
%     DUET, ILD, ITD, Onset Str.
%
% DUET is a binaural blind source seperation algorithm based on a histogram
% of differences in level and phase of the time-frequency representation of
% the mixture signals. We only use the histogram in delta-level-delta-phase
% space and attribute clear peaks to individual sources. This method
% relies on the W-disjunct orthogonality of the sources (e.g. only one
% source is active per time-frequency bin). This is assumption is (mildly)
% violated for echoic mixtures.
%
% [1] Rickard, S. (2007). The DUET Blind Source Separation Algorithm.
%     In S. Makino, H. Sawada, & T.-W. Lee (Eds.),
%     Blind Speech Separation (pp. 217–241). inbook,
%     Dordrecht: Springer Netherlands.
%     http://doi.org/10.1007/978-1-4020-6479-1_8
%
    
    %% PROPERTIES
    
    properties (SetAccess = protected)
        nFreqChannels;  % # of frequency channels
        wSizeSec;       % window size in seconds
        hSizeSec;       % window step size in seconds
        maxDelaySec;    % maximum cross-correleation delay
        maxOffsetDB;    % offset for the onset/offset features on dB
        duetIntegrate;  % will integrate duet features over the whole block
        nSummaryPeaks;  % number of summary peaks
    end
    
    %% METHODS
    
    methods (Access = public)
        
        function obj = FeatureSetNSrcDetection( )
            obj = obj@FeatureCreators.Base();
            obj.nFreqChannels = 16;
            obj.wSizeSec = 0.02;
            obj.hSizeSec = 0.01;
            obj.maxDelaySec = 0.001;
            obj.maxOffsetDB = 30;
            obj.duetIntegrate = true;
        end
        
        function afeRequests = getAFErequests( obj )
            % duet
            % wSTFTSec;   % window size in seconds for STFT
            % wDUETSec;   % window size in seconds for DUET histogram
            % hDUETSec;   % window shift for duet historgram
            % binsAlpha;  % number of histogram bins for alpha dimension
            % binsDelta;  % number of histogram bins for delta dimension
            % maxAlpha;   % masking threshold for alpha dimension
            % maxDelta;   % masking threshold for delta dimension
            afeRequests{1}.name = 'duet';
            afeRequests{1}.params = genParStruct( ...
                'duet_wSTFTSec', obj.wSizeSec,...
                'duet_wDUETSec', (1/2),...
                'duet_hDUETSec', (1/6));
            
            % internaural level differences
            afeRequests{2}.name = 'ild';
            afeRequests{2}.params = genParStruct( ...
                'fb_type', 'gammatone',...
                'fb_nChannels', obj.nFreqChannels,...
                'ihc_method', 'halfwave',... % why is this different for itd and ild?
                'ild_wSizeSec', obj.wSizeSec,...
                'ild_hSizeSec', obj.hSizeSec);
            
            % internaural time differences
            afeRequests{3}.name = 'itd';
            afeRequests{3}.params = genParStruct( ...
                'cc_wSizeSec', obj.wSizeSec,...
                'cc_hSizeSec', obj.hSizeSec,...
                'cc_maxDelaySec', obj.maxDelaySec,...
                'pp_bMiddleEarFiltering', true,...
                'pp_bNormalizeRMS', true,...
                'ihc_method', 'dau',... % why is this different for itd and ild?
                'fb_type', 'gammatone',...
                'fb_nChannels', obj.nFreqChannels);
            
            % onset strengths
            afeRequests{4}.name = 'onsetStrength';
            afeRequests{4}.params = genParStruct( ...
                'pp_bNormalizeRMS', false, ...
                'ons_maxOffsetdB', obj.maxOffsetDB,...
                'ofs_maxOffsetdB', obj.maxOffsetDB,...
                'fb_nChannels', obj.nFreqChannels);

            % spectral features
            afeRequests{5}.name = 'spectralFeatures';
            afeRequests{5}.params = genParStruct( ...
                'pp_bNormalizeRMS', false, ...
                'fb_nChannels', obj.nFreqChannels);
        end
        
        function x = constructVector( obj )
            % constructVector from afe requests
            %   #1: DUET, #2: ILD, #3: ITD, #4: OnS, #5: Spectral
            %
            %   See getAFErequests
            
            x = {};
            
            % afeIdx 1: STFT -> DUET histogram + summary
            duet = obj.afeData(1);
            duet_hist = obj.makeDuetFeatureBlock(...
                duet{1},...
                ~obj.descriptionBuilt,...
                obj.duetIntegrate);
            duet_hist = obj.reshapeBlock(duet_hist, 1);
            duet_hist = obj.reshape2featVec(duet_hist);
            x = obj.concatFeats(x, duet_hist);
            % duet histogram summary
            duet_summ = obj.makePeakSummaryFeatureBlock(...
                duet_hist{1},...
                ~obj.descriptionBuilt,...
                7, 4);
            duet_summ = obj.reshape2featVec(duet_summ);
            x = obj.concatFeats(x, duet_summ);
            
            % afeIdx 2+3: ILD/ITD LMoments per freq-channel
            ild = obj.makeBlockFromAfe(2, 1, ...
                @(a)(a.Data), ...
                {@(a)(a.Name),@(a)([num2str(numel(a.cfHz)) '-ch']),@(a)(a.Channel)}, ...
                {@(a)(strcat('t', arrayfun(@(t)(num2str(t)),1:size(a.Data,1),'UniformOutput',false)))}, ...
                {@(a)(strcat('f', arrayfun(@(f)(num2str(f)),a.cfHz,'UniformOutput',false)))} );
            ildLM = obj.block2feat(...
                ild,...
                @(b)(lMomentAlongDim( b, [1,2,3,4], 1, true )),...
                2,...
                @(idxs)(sort([idxs idxs idxs idxs])),...
                {{'1.LMom',@(idxs)(idxs(1:4:end))},...
                 {'2.LMom',@(idxs)(idxs(2:4:end))},...
                 {'3.LMom',@(idxs)(idxs(3:4:end))},...
                 {'4.LMom',@(idxs)(idxs(4:4:end))}} ); 
            x = obj.concatFeats(x, ildLM);
            itd = obj.makeBlockFromAfe(3, 1, ...
                @(a)(a.Data), ...
                {@(a)(a.Name),@(a)([num2str(numel(a.cfHz)) '-ch']),@(a)(a.Channel)}, ...
                {@(a)(strcat('t', arrayfun(@(t)(num2str(t)),1:size(a.Data,1),'UniformOutput',false)))}, ...
                {@(a)(strcat('f', arrayfun(@(f)(num2str(f)),a.cfHz,'UniformOutput',false)))} );
            itdLM = obj.block2feat(...
                itd,...
                @(b)(lMomentAlongDim(b, [1,2,3,4], 1, true)),...
                2,...
                @(idxs)(sort([idxs idxs idxs idxs])),...
                {{'1.LMom',@(idxs)(idxs(1:4:end))},...
                 {'2.LMom',@(idxs)(idxs(2:4:end))},...
                 {'3.LMom',@(idxs)(idxs(3:4:end))},...
                 {'4.LMom',@(idxs)(idxs(4:4:end))}} ); 
            x = obj.concatFeats(x, itdLM);
            
            % afeIdx 2+3: ILD + ITD 2D histogram, peak summary + LMoments
            itd_ild_hist_data = histcounts2(ild{1}, itd{1}, 'Normalization', 'probability');
            itd_ild_hist_data = itd_ild_hist_data./max(max(itd_ild_hist_data));
            itd_ild_hist_summ = obj.makePeakSummaryFeatureBlock(...
                itd_ild_hist_data,...
                ~obj.descriptionBuilt,...
                7, 4);
            itd_ild_hist_summ = obj.reshape2featVec(itd_ild_hist_summ);
            x = obj.concatFeats(x, itd_ild_hist_summ);
            
            % afeIdx 4: onsetStrengths LMoments
            onsR = obj.makeBlockFromAfe( 4, 1,...
                @(a)(compressAndScale( a.Data, 0.33 )),...
                {@(a)(a.Name),@(a)([num2str(numel(a.cfHz)) '-ch']),@(a)(a.Channel)},...
                {'t'},...
                {@(a)(strcat('f', arrayfun(@(f)(num2str(f)), a.cfHz, 'UniformOutput', false)))} );
            onsL = obj.makeBlockFromAfe( 4, 2,...
                @(a)(compressAndScale( a.Data, 0.33 )),...
                {@(a)(a.Name),@(a)([num2str(numel(a.cfHz)) '-ch']),@(a)(a.Channel)},...
                {'t'},...
                {@(a)(strcat('f', arrayfun(@(f)(num2str(f)), a.cfHz, 'UniformOutput', false)))} );
            ons = obj.combineBlocks( @(b1,b2)(0.5*b1+0.5*b2), 'LRmean', onsR, onsL );
            onsLM = obj.block2feat(...
                ons,...
                @(b)(lMomentAlongDim( b, [1,2,3,4], 1, true )),...
                2,...
                @(idxs)(sort([idxs idxs idxs idxs])),...
                {{'1.LMom',@(idxs)(idxs(1:4:end))},...
                 {'2.LMom',@(idxs)(idxs(2:4:end))},...
                 {'3.LMom',@(idxs)(idxs(3:4:end))},...
                 {'4.LMom',@(idxs)(idxs(4:4:end))}} );
             x = obj.concatFeats( x, onsLM );
             
            % afeIdx 5: spectralFeatures
            spfR = obj.makeBlockFromAfe( 5, 1, ...
                @(a)(compressAndScale( a.Data, 0.33 )), ...
                {@(a)(a.Name),[num2str(obj.nFreqChannels) '-ch'], ...
                @(a)(a.Channel)}, ...
                {'t'}, ...
                {@(a)(a.fList)} );
            spfL = obj.makeBlockFromAfe( 5, 2, ...
                @(a)(compressAndScale( a.Data, 0.33 )), ...
                {@(a)(a.Name),[num2str(obj.nFreqChannels) '-ch'],...
                @(a)(a.Channel)}, ...
                {'t'}, ...
                {@(a)(a.fList)} );
            spf = obj.combineBlocks( @(b1,b2)(0.5*b1+0.5*b2), 'LRmean', spfR, spfL );
            spgLM = obj.block2feat(...
                spf,...
                @(b)(lMomentAlongDim( b, [1,2,3,4], 1, true )),...
                2,...
                @(idxs)(sort([idxs idxs idxs idxs])),...
                {{'1.LMom',@(idxs)(idxs(1:4:end))},...
                 {'2.LMom',@(idxs)(idxs(2:4:end))},...
                 {'3.LMom',@(idxs)(idxs(3:4:end))},...
                 {'4.LMom',@(idxs)(idxs(4:4:end))}} );
             x = obj.concatFeats( x, spgLM );
        end
        
        function outputDeps = getFeatureInternOutputDependencies( obj )
            % relevant members
            outputDeps.nFreqChannels = obj.nFreqChannels;
            outputDeps.wSizeSec = obj.wSizeSec;
            outputDeps.hSizeSec = obj.hSizeSec;
            outputDeps.maxDelaySec = obj.maxDelaySec;
            outputDeps.maxOffsetDB = obj.maxOffsetDB;
            outputDeps.duetIntegrate = obj.duetIntegrate;
            % classname
            classInfo = metaclass( obj );
            classnames = strsplit( classInfo.Name, '.' );
            outputDeps.featureProc = classnames{end};
            % version
            outputDeps.v = 35;
        end
        
    end
    
    %% STATIC METHODS
    
    methods (Static)
        
        function rval = makeDuetFeatureBlock(...
            data_in,...
            request_description,...
            integrate_time)
            %createDuetFeatureBlock   creates valid block with description (optional)
            %
            % INPUTS:
            %   duet                : histogram input
            %   requestDescription  : if true, build block annotations
            %   do_integration      : if true, integrate the stream in to one histogram
            %
            
            % init
            if nargin < 1
                error('method needs at least the histogram as input!');
            end
            if nargin < 2
                request_description = false;
            end
            if nargin < 3
                integrate_time = false;
            end
            data = data_in.Data;
            nFrames = size(data, 1);
            
            % build histgram block
            if integrate_time
                data = mean(data,1);
                nFrames = 1;
            end
            rval = { data };
            if request_description
                % build histogram block description
                histGrpInfo = {'duet_hist',...
                               [num2str(nFrames) 'x' ...
                                num2str(data_in.binsAlpha) 'x' ...
                                num2str(data_in.binsDelta) '-hist'],...
                               'mono'};
                timeAxisVal = arrayfun(@(a)(['t' num2str(a)]),1:nFrames,'UniformOutput',false);
                alphaAxisVal = arrayfun(@(a)(['a' num2str(a)]),...
                    linspace(-data_in.maxAlpha, data_in.maxAlpha, data_in.binsAlpha),...
                    'UniformOutput',false);
                deltaAxisVal = arrayfun(@(a)(['d' num2str(a)]),...
                    linspace(-data_in.maxDelta, data_in.maxDelta, data_in.binsDelta),...
                    'UniformOutput',false);
                for ii = 1:nFrames
                    timeInfo{ii} = { histGrpInfo{:}, timeAxisVal{ii} };
                end
                for ii = 1:data_in.binsAlpha
                    alphaInfo{ii} = { histGrpInfo{:}, alphaAxisVal{ii} };
                end
                for ii = 1:data_in.binsDelta
                    deltaInfo{ii} = { histGrpInfo{:}, deltaAxisVal{ii} };
                end
                rval = { rval{:}, timeInfo, alphaInfo, deltaInfo };
            end
        end
        
        function rval = makePeakSummaryFeatureBlock(...
            data,...
            request_description,...
            n_summary_peaks,...
            n_l_moments)
            %makePeakSummaryFeatureBlock   creates a summary block from 2d histgram
            %   containing the highest peaks and the first LMoments of the histogram
            %
            % INPUTS:
            %   data                : histogram input
            %   requestDescription  : if true, build block annotations
            %   n_summary_peaks     : # of peaks in the summary
            %   n_l_moments         : # of LMoments to compute
            %
            
            % init
            if nargin < 1
                error('method needs at least the histogram as input!');
            end
            if nargin < 2
                request_description = false;
            end
            if nargin < 3
                n_summary_peaks = 10;
            end
            if nargin < 4
                n_l_moments = 10;
            end
            
            % build summary block
            summaryData = zeros(1, n_summary_peaks+n_l_moments);
            try
                frame_peaks = extrema2(data);
            catch
                frame_peaks = zeros(1,n_summary_peaks);
            end
            for ii = 1:min(n_summary_peaks,numel(frame_peaks))
                try
                    summaryData(ii) = frame_peaks(ii);
                catch
                    % pass
                end
            end
            data_flat = reshape(data, numel(data), 1);
            for ii = 1:n_l_moments
                try
                    summaryData(n_summary_peaks+ii) = lMoments(data_flat, ii);
                catch
                    % pass
                end
            end
            rval = { summaryData };
            if request_description
                % build summary block desctiption
                grpInfo = {'duet_summary',...
                           ['1x' num2str(n_summary_peaks+n_l_moments)],...
                           'mono'};
                for ii = 1:n_summary_peaks
                    summaryInfo{ii} = {grpInfo{:}, [num2str(ii) '.Peak']};
                end
                for ii = 1:n_l_moments
                    summaryInfo{n_summary_peaks+ii} = {grpInfo{:}, [num2str(ii) '.LMom']};
                end
                rval = { rval{:}, {{ grpInfo{:}, 't1' }}, summaryInfo };
            end
        end
        
    end
    
end
