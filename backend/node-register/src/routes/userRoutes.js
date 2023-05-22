const express = require('express');
const userController = require('../controllers/userController');
const authMiddleware = require('../middlewares/authMiddleware');

const router = express.Router();

router.post('/register',  userController.create);
router.put('/:id', userController.update);
router.get('/:id', userController.findById);
router.get('/', userController.findAll);

module.exports = router;
