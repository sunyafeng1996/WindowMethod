classdef DataSet
    % 处理训练数据集
    
    properties
        samples
        min_samples
        max_samples
        samples_normalized
        samples_normalized_dl
        
        labels
        min_labels
        max_labels
        labels_normalized
        labels_normalized_dl
    end
    
    methods
        function obj = DataSet(Population,C)
            obj.samples=Population.decs;
            if C==0
                obj.labels=Population.objs;
            else
                obj.labels=[Population.objs,Population.cons];
            end
            [obj.min_samples, ~] = min(obj.samples(:));
            [obj.max_samples, ~] = max(obj.samples(:));
            obj.samples_normalized = (obj.samples - obj.min_samples) / (obj.max_samples - obj.min_samples+1e-4);
            [obj.min_labels, ~] = min(obj.labels(:));
            [obj.max_labels, ~] = max(obj.labels(:));
            obj.labels_normalized = (obj.labels - obj.min_labels) / (obj.max_labels - obj.min_labels+1e-4);
            % obj.samples_normalized_dl=dlarray(obj.samples_normalized','CB');
            % obj.labels_normalized_dl=dlarray(obj.labels_normalized','CB');
            X=obj.samples_normalized';
            X = reshape(X, [size(Population.decs,2), 1, 1, size(Population.decs,1)]);
            obj.samples_normalized_dl=dlarray(X,'SSCB');
            Y = obj.labels_normalized';
            obj.labels_normalized_dl=dlarray(Y,'CB');

        end
        
        function data = recover_samples(obj,data_dl,tag)
            datas=extractdata(data_dl);
            datas=datas';
            if tag=="samples"
                maxd=obj.max_samples;
                mind=obj.min_samples;
            else
                maxd=obj.max_labels;
                mind=obj.min_labels;
            end
            data = datas * (maxd - mind+1e-4) + mind;
        end
    end
end

