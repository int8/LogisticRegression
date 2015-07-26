function accuracy(labels, predictions) 
   sum(predictions .== labels) / length(labels)
end 

function sensitivity(predictions, labels) 
    tp = 0 
    fptp = 0. 
    for i in 1:length(labels)
      if labels[i] == 1 
          tp += predictions[i]
          fptp += 1.  
      end    
 end 
    return tp / fptp 
end

function specificity(predictions, labels) 
    tn = 0 
    tnfp = 0. 
    for i in 1:length(labels)
      if labels[i] == 0 
          tn += 1 - predictions[i]
          tnfp += 1.  
      end    
    end 
    return tn / tnfp 
end

function precision(predictions, labels) 
    tp = 0 
    fp = 0. 
    for i in 1:length(labels)
      if labels[i] == 1 
          tp += predictions[i]
      else 
          fp += predictions[i]
      end    
    end 
    return tp / (tp + fp)
end

function triangle_area(A,B,C) 
  0.5 * abs((A[1] - C[1]) * (B[2] - A[2]) - (A[1] - B[1]) * ( C[2] - A[2]))
end

function auc(labels, predictions) 
   n = length(labels)
   ROC_curve_points = ones(n+1,2)
   sorted_labels = labels[sortperm(predictions)];
   prev_point = [1., 1.];
   total_auc = 0.
   predicted_labels = ones(n)
   for i in 1:n
     predicted_labels[i] = 0 
     A = [1 - specificity(predicted_labels,sorted_labels), sensitivity(predicted_labels,sorted_labels) ]     
     ROC_curve_points[i,:] = A
	 B = prev_point
     C = [1 0] 
     total_auc += triangle_area(A,B,C)     
     prev_point = copy(A)
   end    
   return ROC_curve_points, total_auc
end


function cross_validation(k, learn_method, dataset, labels, evaluation_metric)
  n = size(dataset,1)
  if n < k
     error("nr of folds is greater than number of observations")
  end
  # shuffle dataset
  shuffled_indices = shuffle([1:n])
  # divide dataset into k chunks
  fold_cardinality = div(n,k)
  fold_indices = [[(fold_cardinality *(i-1) + 1):(fold_cardinality *i)] for i in 1:k]
  # add mising observations to the last chunk
  fold_indices[end] = fold_indices[end][end] < n? vcat(fold_indices[end],(fold_indices[end][end]+1):n) : fold_indices[end]

  f = learn_method[:learning_method] 
  infer = learn_method[:inferencer]
  params = learn_method[:params]
  
  predictions = zeros(n)
  
  for fold = fold_indices
      test_indices = shuffled_indices[fold]
      training_indices = filter(a -> !(a in test_indices), shuffled_indices)
      training_dataset = dataset[training_indices,:]
      training_labels = labels[training_indices]
      
	  model_params = f(training_dataset, training_labels, params)
	  
      test_dataset = dataset[test_indices ,:]    	  
	  predictions[test_indices] = infer(test_dataset, model_params)[:]	  
	  
  end
  
  return predictions, evaluation_metric(labels, predictions)
    
end

