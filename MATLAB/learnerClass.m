classdef learnerClass
    properties (Access = public)
      Ws    % memorized concepts
      Xs    % memorized teaching examples
      Ys    % memorized teaching examples labels
      Ysl   % learner provided label
      Ysl_prob % probability of learner provide label (Teacher's assets)
      beta  % memory decay rate
      order % teaching sequence index (COULD HAVE DUPLICATE ONES)
    end
    methods
        function obj = learnerClass(beta, w0)
            obj.Ws = w0;
            obj.Xs = [];
            obj.Ys = [];
            obj.Ysl = [];
            obj.Ysl_prob = [];
            obj.beta = beta;
            obj.order = [];
        end
    end
end