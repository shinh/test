(define (gen-line n c)
  (if (> n 0)
      (cons (gen-line (- n 1) c) c) c))

(define (gen-tree-aux n m)
  (if (> n 0)
      (let ((c (gen-tree-aux (- n 1) m)))
        (cons (gen-line m c) c))
      (gen-line m 1)))

(define (gen-tree n) (gen-tree-aux n n))

(define (output-graph nodetab)
  (with-output-to-file "tree.dot" 
    (lambda ()
      (format #t "digraph G {")
      (let ((no 0)
	    (pair2no (make-hash-table)))
	(hash-table-for-each 
	 nodetab 
	 (lambda (from tos)
	   (let ((fromno (cond ((hash-table-get pair2no from #f)
				(hash-table-get pair2no from))
			       (else
				(inc! no 1)
				(hash-table-put! pair2no from no)
				no))))
	     (map (lambda (to)
		    (cond
		     ((pair? to)
		      (let ((tono (cond ((hash-table-get pair2no to #f)
					 (hash-table-get pair2no to))
					(else
					 (inc! no 1)
					 (hash-table-put! pair2no to no)
					 no))))
			(format #t "\"node~s\" -> \"node~s\"\n" fromno tono)))
		     (else
		      (format #t "\"node~s\" -> \"~s\"\n" fromno to))))
		  tos)))))
    (format #t "}"))))

				 

(define (traverse-tree parent tree nodetab)
  (hash-table-push! nodetab parent tree)
  (cond
   ((hash-table-get nodetab tree #f))
   ((pair? tree)
    (traverse-tree tree (car tree) nodetab)
    (traverse-tree tree (cdr tree) nodetab))))

(define (tree-graph tree)
  (let ((nodetab (make-hash-table)))
    (traverse-tree :top tree nodetab)
    (output-graph nodetab)))
 
;(tree-graph '(a b (c)))   
(tree-graph (gen-tree 1))
