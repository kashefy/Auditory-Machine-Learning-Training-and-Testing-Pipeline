function boxplot_grps( labels, cgroup, nvPairs, varargin )

grps = cell( size( varargin ) );
cgrps = cell( size( varargin ) );
setlabels = isempty( labels );
for ii = 1 : numel( varargin )
    grps{ii} = ii * ones( size( varargin{ii} ) );
    if ~isempty( cgroup ), cgrps{ii} = cgroup(ii) * ones( size( varargin{ii} ) ); end
    if setlabels
        labels{ii} = inputname( ii + 3 );
    end
    labels{ii} = [labels{ii} '  (' num2str(numel(varargin{ii})) ') '];
end
%if isempty( cgroup ), cgrps = gt; end
    
if isempty( nvPairs )
    nvPairs = {
         'notch', 'on', ...
         'whisker', inf, ...
         'widths', 0.8,...
         };
end
    
boxplot( [varargin{:}], ...
         [grps{:}], ...
         'labels', labels,...
         'colorgroup', [cgrps{:}],...
         'labelorientation', 'inline',...
         nvPairs{:} ...
         )

