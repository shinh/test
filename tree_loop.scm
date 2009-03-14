(define cnt 0)
(define (no-cycle?-aux tr tr2 fst)
  (set! cnt (+ cnt 1))
  (if (and (not fst) (eq? tr tr2) (not (null? tr)))
      #f
      (if (and (pair? tr2) (pair? tr))
	  (and
	   (if (pair? (car tr2))
	       (and
		(no-cycle?-aux (car tr) (car (car tr2)) #f)
		(no-cycle?-aux (cdr tr) (car (car tr2)) #f)
		(no-cycle?-aux (car tr) (cdr (car tr2)) #f)
		(no-cycle?-aux (cdr tr) (cdr (car tr2)) #f))
	       #t)
	   (if (pair? (cdr tr2))
	       (and
		(no-cycle?-aux (car tr) (car (cdr tr2)) #f)
		(no-cycle?-aux (cdr tr) (car (cdr tr2)) #f)
		(no-cycle?-aux (car tr) (cdr (cdr tr2)) #f)
		(no-cycle?-aux (cdr tr) (cdr (cdr tr2)) #f))
	       #t))
	  #t)))

(define (no-cycle? tr)
  (no-cycle?-aux tr tr #t))

;; (print (no-cycle? '(a b x)))
;; (print (no-cycle? '((a b) x c)))

;; (define n '((a b) x c))
;; (set-cdr! (cdr (car n)) (car n))
;; (print (no-cycle? n))

;; (define m '(a (a b) x c))
;; (set-car! m m)
;; (print (no-cycle? n))

;; (print (no-cycle? '(1 (2 (3 4) ()) ())))
;; (print (no-cycle? '((a . #1=(c . d)) #1#)))
;; (print (no-cycle? '(#1=(a (b (c d) e) (f (g h) (i (j (k l) m)))
;; 			  (n (o p) (q r)) s (t (u v) (w x (y z #1#)))))))
;; (print (no-cycle? '(#1=(a (b (c d) e) (f (g h) (i (j (k l) m)))
;; 			  (n (o p) (q r)) s (t (u v) (w x (y z)))))))

(define (gen-line n c)
  (if (> n 0)
      (cons (gen-line (- n 1) c) c) c))

(print (gen-line 3 1))

(define (gen-tree-aux n m)
  (if (> n 0)
      (let ((c (gen-tree-aux (- n 1) m)))
        (cons (gen-line m c) c))
      (gen-line m 1)))

(define (gen-tree n) (gen-tree-aux n n))

(define (run n)
  (if (> n 0) (let ()
                (run (- n 1))
                (no-cycle? (gen-tree n))
                (print cnt)
                (set! cnt 0))))
(run 25)
